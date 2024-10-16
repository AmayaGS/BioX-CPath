import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from models.KRAG_heatmap_model import KRAG_Classifier
from .heatmap_generator import HeatmapGenerator
from .graph_visualiser import GraphVisualiser
from .graph_metrics import GraphMetricGenerator
from .metrics_visualiser import MetricsVisualiser
from train_test_loops.krag_heatmap_loop import heatmap_scores

class KRAGResultsGenerator:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_data(self):
        graph_dict = self._load_graph_dict()

        if self.args.specific_ids and self.args.test_fold:
            self.logger.info(f"Processing specific patients for {self.args.test_fold}")
            self._process_fold(self.args.test_fold, self.args.specific_ids, graph_dict)
        else:
            self.logger.warning("No specific patient IDs or test fold provided. Processing all patients in all folds.")
            splits = self._load_splits()
            for fold, fold_data in enumerate(splits.items()):
                self.logger.info(f"Processing fold: {fold}")
                self._process_fold(fold, fold_data[1]['Test'], graph_dict)

    def _process_fold(self, fold, patient_ids, graph_dict):
        graph_net = self._initialize_model(fold)
        model_name = self.results_dir.split('/')[-1]
        vis_path = os.path.join(self.args.directory, "vis_data", model_name)
        output_dir = os.path.join(self.args.directory, "graph_visualisations", model_name)
        os.makedirs(vis_path, exist_ok=True)
        fold_dir = os.path.join(output_dir, f"Fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        all_graphs = {}
        all_preds = {}
        all_metrics = {}
        graph_visualiser = GraphVisualiser(self.args, self.results_dir, self.logger)
        heatmap_generator = HeatmapGenerator(self.args, self.results_dir, self.logger)
        metrics_generator = GraphMetricGenerator(self.args, self.results_dir, self.logger)
        metrics_visualiser = MetricsVisualiser()

        for patient_id in patient_ids:
            if patient_id not in graph_dict:
                self.logger.info(f"Warning: Patient ID {patient_id} not found in graph dictionary. Skipping.")
                continue

            data_path = os.path.join(vis_path, f"patient_data_{self.args.graph_mode}_fold_{fold}_{patient_id}_{self.args.dataset_name}.pkl")

            if os.path.exists(data_path):
                self.logger.info(f"Loading existing data for patient: {patient_id}")
                with open(data_path, 'rb') as f:
                    patient_data = pickle.load(f)
            else:
                self.logger.info(f"Calculating data for patient: {patient_id}")
                patient_data = self._calculate_patient_data(self.args, patient_id, graph_dict, graph_net)
                with open(data_path, 'wb') as f:
                    pickle.dump(patient_data, f)

            patient_dir = os.path.join(fold_dir, f"{patient_id}")
            os.makedirs(patient_dir, exist_ok=True)
            # Visualize graphs
            graph_visualiser.visualise_graphs(patient_id, patient_dir, patient_data, fold)
            # Generate heatmaps
            heatmap_generator.generate_heatmaps(patient_id, patient_dir, patient_data['attention_scores'], fold)
            # Generate metrics
            metrics_path = data_path[:-4] + '_metrics.pkl'
            if os.path.exists(metrics_path):
                self.logger.info(f"Loading existing metrics for patient: {patient_id}")
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
            else:
                self.logger.info(f"Calculating metrics for patient: {patient_id}")
                metrics = metrics_generator.generate_metrics(patient_id, patient_dir, patient_data['graphs'],
                                                         patient_data['actual_label'], fold)
                with open(metrics_path, 'wb') as f:
                    pickle.dump(metrics, f)

            all_graphs[patient_id] = patient_data['graphs']
            all_preds[patient_id] = [patient_data['actual_label'], patient_data['predicted_label']]
            all_metrics[patient_id] = metrics

        self._save_pred_results(all_preds, fold)
        metrics_visualiser.plot_metrics(all_metrics, fold_dir)

        # # global edge weight distribution
        # if len(all_graphs) > 1:
        #     graph_visualiser.plot_global_edge_weight_distribution(fold_dir, all_graphs, fold)
        # else:
        #     self.logger.info("Skipping global edge weight distribution plot as only one patient is processed.")

    def _calculate_patient_data(self, patient_id, graph_dict, graph_net):
        slide_embedding = graph_dict[patient_id]
        test_graph_loader = DataLoader(slide_embedding, batch_size=1, shuffle=False,
                                       num_workers=self.args.num_workers)

        actual_label, predicted_label, metadata, attention_scores, layer_data = heatmap_scores(self.args,
            graph_net, test_graph_loader, patient_id, self.loss_fn, n_classes=self.args.n_classes
        )

        patient_graphs = self._create_patient_graphs(layer_data, patient_id)

        return {
            'actual_label': actual_label.item(),
            'predicted_label': predicted_label.item(),
            'metadata': metadata,
            'attention_scores': attention_scores,
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
        node_scores = 100 + (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min()) * 500
        edge_weights = data.attention_weights.cpu().detach().numpy().mean(axis=1)
        edge_weights = 1 + (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min()) * 9

        for j, node in enumerate(G.nodes()):
            original_index = data.node_mapping[j]
            G.nodes[node]['stain_type'] = data.node_attr[original_index].item()
            G.nodes[node]['filename'] = data.node_filenames[j]
            G.nodes[node]['score'] = node_scores[j]

        for (u, v, edge_data) in G.edges(data=True):
            original_u = data.node_mapping[u]
            original_v = data.node_mapping[v]
            original_edge = str([original_u, original_v])
            original_edge_index = data.edge_mapping[original_edge]
            edge_type = data.original_edge_attr[original_edge_index].item()
            edge_data['edge_attribute'] = edge_type
            attention_idx = data.edge_idx_mapping[original_edge_index]
            edge_data['weight'] = edge_weights[attention_idx].item()

        return G

    def _initialize_model(self, fold):
        graph_net = KRAG_Classifier(
            self.args.embedding_vector_size,
            hidden_dim=self.args.hidden_dim,
            num_classes=self.args.n_classes,
            heads=self.args.heads,
            pooling_ratio=self.args.pooling_ratio,
            walk_length=self.args.encoding_size,
            conv_type=self.args.convolution,
            attention=self.args.attention
        )

        checkpoints = os.path.join(self.results_dir, "checkpoints", "best_val_models")
        checkpoint = os.path.join(checkpoints, f"checkpoint_fold_{fold}_accuracy.pth")
        checkpoint_weights = torch.load(checkpoint)
        graph_net.load_state_dict(checkpoint_weights)
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