import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import gc

from mains.main_embedding import use_gpu
from models.BioXCPath_explainability import BioXCPath_explainable_model
from .graph_metrics import GraphMetricGenerator
from train_test_loops.krag_heatmap_loop import heatmap_scores


class VisualisationResultsGenerator:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()

    def process_fold(self, fold, patient_ids):
        """
        Process a single fold and generate metrics for visualization.

        Args:
            fold: Fold number
            patient_ids: List of patient IDs to process

        Returns:
            dict: Dictionary of metrics data for all processed patients
        """
        graph_dict = self._load_graph_dict()
        graph_net = self._initialize_model(fold)

        all_metrics = {}
        all_patient_data = {}

        for patient_id in patient_ids:
            if patient_id not in graph_dict:
                self.logger.info(f"Warning: Patient ID {patient_id} not found in graph dictionary. Skipping.")
                continue

            try:
                # Generate data path
                data_path = os.path.join(
                    self.args.directory,
                    "vis_data",
                    os.path.basename(self.results_dir),
                    f"patient_data_{self.args.graph_mode}_fold_{fold}_{patient_id}_{self.args.dataset_name}"
                )

                # Load or calculate patient data
                patient_data = self._get_patient_data(data_path, patient_id, graph_dict, graph_net)
                if patient_data is None:
                    continue

                all_patient_data[patient_id] = patient_data

                # Process metrics
                metrics_path = f"{data_path}_graph_metrics.pkl"
                metrics = self._get_patient_metrics(metrics_path, patient_id, patient_data)
                if metrics is None:
                    continue

                # Combine all patient data for visualization
                all_metrics[patient_id] = [
                    patient_data['actual_label'],
                    patient_data['predicted_label'],
                    patient_data['stain_attention'],
                    patient_data['layer_attention'],
                    patient_data['entropy_scores'],
                    metrics
                ]

            except Exception as e:
                self.logger.error(f"Error processing patient {patient_id}: {e}")
                continue

        return all_metrics, all_patient_data

    def _get_patient_data(self, data_path, patient_id, graph_dict, graph_net):
        """Load or calculate patient data."""
        if os.path.exists(f"{data_path}.pkl"):
            self.logger.info(f"Loading existing data for patient: {patient_id}")
            with open(f"{data_path}.pkl", 'rb') as f:
                return pickle.load(f)

        self.logger.info(f"Calculating data for patient: {patient_id}")
        try:
            torch.cuda.empty_cache()
            gc.collect()
            patient_data = self._calculate_patient_data(patient_id, graph_dict, graph_net)
            with open(f"{data_path}.pkl", 'wb') as f:
                pickle.dump(patient_data, f)
            return patient_data
        except Exception as e:
            self.logger.error(f"Error calculating data for patient {patient_id}: {e}")
            return None

    def _get_patient_metrics(self, metrics_path, patient_id, patient_data):
        """Load or calculate patient metrics."""
        if os.path.exists(metrics_path):
            self.logger.info(f"Loading existing metrics for patient: {patient_id}")
            with open(metrics_path, 'rb') as f:
                return pickle.load(f)

        self.logger.info(f"Calculating metrics for patient: {patient_id}")
        try:
            metrics_generator = GraphMetricGenerator(self.args, self.results_dir, self.logger)
            metrics = metrics_generator.generate_metrics(
                patient_id, None, patient_data['graphs'],
                patient_data['actual_label'], None
            )
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics for patient {patient_id}: {e}")
            return None

    def _calculate_patient_data(self, patient_id, graph_dict, graph_net):
        slide_embedding = graph_dict[patient_id]
        test_graph_loader = DataLoader(slide_embedding, batch_size=1, shuffle=False,
                                       num_workers=self.args.num_workers, pin_memory=False)

        actual_label, predicted_label, metadata, attention_scores, layer_data, layer_attention, stain_attention, entropy_scores = heatmap_scores(self.args,
            graph_net, test_graph_loader, patient_id, self.loss_fn, n_classes=self.args.n_classes)

        patient_graphs = self._create_patient_graphs(layer_data, patient_id)

        return {
            'actual_label': actual_label.item(), # this might not work anymore
            'predicted_label': predicted_label.item(),
            'metadata': metadata,
            'attention_scores': attention_scores,
            'layer_attention': layer_attention,
            'stain_attention': stain_attention,
            'entropy_scores': entropy_scores,
            'layer_data': layer_data,
            'graphs': patient_graphs
        }

    def _create_patient_graphs(self, layer_data, patient_id):
        patient_graphs = []
        for layer in layer_data[patient_id]:
            G = self._create_networkx_graph(layer)
            patient_graphs.append(G)
        return patient_graphs


    def _create_networkx_graph(self, data):
        # Convert torch geometric data to networkx graph. THIS DOES NOT CONSERVE THE EDGE AND NODE INDEXES!!!
        G = to_networkx(data, to_undirected=False)

        node_scores = data.node_att_scores.cpu().detach().numpy()
        node_scores = 100 + (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-9) # Normalizing node scores
        edge_weights = data.attention_weights.cpu().detach().numpy().mean(axis=1)
        edge_weights = 1 + (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min()) # Normalizing edge weights

        for j, node in enumerate(G.nodes()):
            original_index = data.node_mapping[j]
            G.nodes[node]['stain_type'] = data.node_attr[original_index].item() # Mapping back to original node index to get the stain type
            G.nodes[node]['filename'] = data.node_filenames[j]
            G.nodes[node]['score'] = node_scores[j]

        for (u, v, edge_data) in G.edges(data=True):
            original_u = data.node_mapping[u] # source node
            original_v = data.node_mapping[v] # target node
            original_edge = str([original_u, original_v])
            original_edge_index = data.edge_mapping[original_edge] # Mapping back to original edge index to get the edge type
            edge_type = data.original_edge_attr[original_edge_index].item()
            edge_data['edge_attribute'] = edge_type
            attention_idx = data.edge_idx_mapping[original_edge_index]
            edge_data['weight'] = edge_weights[attention_idx].item()

        return G

    def _initialize_model(self, fold):
        graph_net = BioXCPath_explainable_model(in_features=self.args.embedding_vector_size, edge_attr_dim=len(self.args.edge_types),
                                                node_attr_dim=len(self.args.stain_types), hidden_dim=self.args.hidden_dim,
                                                num_classes=self.args.n_classes, heads=self.args.heads, pooling_ratio=self.args.pooling_ratio,
                                                walk_length=self.args.encoding_size, conv_type=self.args.convolution,
                                                num_layers=self.args.num_layers, embedding_dim=10, dropout_rate=self.args.dropout,
                                                use_node_embedding=self.args.use_node_embedding, use_edge_embedding=self.args.use_edge_embedding,
                                                use_attention=self.args.use_attention)

        checkpoints = os.path.join(self.results_dir, "checkpoints", "best_val_models")
        checkpoint = os.path.join(checkpoints, f"checkpoint_fold_{fold}_accuracy.pth")
        checkpoint_weights = torch.load(checkpoint, weights_only=True)
        graph_net.load_state_dict(checkpoint_weights, strict=False)
        if torch.cuda.is_available():
            graph_net.cuda()
        graph_net.eval()
        return graph_net

    def _save_pred_results(self, patient_data, fold):
        results = [(patient_id, actual_label, predicted_label)
                   for patient_id, (actual_label, predicted_label) in patient_data.items()]
        df_results = pd.DataFrame(results, columns=['Patient_ID', 'Label', 'Predicted_label'])

        df_results.to_csv(
            self.results_dir + f"/predicted_results_{self.args.graph_mode}_{self.args.dataset_name}_fold_{fold}.csv",
            index=False)

    def _load_splits(self):
        with open(self.args.directory + f"/train_test_strat_splits_{self.args.dataset_name}.pkl", "rb") as file:
            return pickle.load(file)

    def _load_graph_dict(self):
        graph_dict_path = self._get_graph_dict_path()
        with open(graph_dict_path, "rb") as file:
            return pickle.load(file)

    def _get_graph_dict_path(self):
        base_path = self.args.directory + f"/dictionaries/{self.args.graph_mode}_dict_{self.args.dataset_name}"
        if self.args.encoding_size > 0:
            base_path += f"_positional_encoding_{self.args.encoding_size}"
        return base_path + f"_{self.args.embedding_net}_{self.args.stain_type}.pkl"