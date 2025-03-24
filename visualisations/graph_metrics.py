import numpy as np
from collections import defaultdict

# nested dictionary to store all metrics (see schema at bottom of file)
class GraphMetricGenerator:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.stain_types_rev = {v: k for k, v in self.args.stain_types.items()}
        self.edge_types_rev = {v: k for k, v in self.args.edge_types.items()}

    def generate_metrics(self, patient_id, patient_dir, graphs, label, fold):
        metrics = defaultdict(list)
        metrics[patient_id].append({'Label': label})
        layers = {}

        for layer, (G) in enumerate(graphs, 1):
            properties = self.calculate_graph_properties(G, patient_id, label, layer, patient_dir, fold)
            layers[f"Layer_{layer}"] = properties
        metrics[patient_id].append(layers)

        return metrics

    def calculate_graph_properties(self, G, patient_id, label, layer, fold_dir, fold):
        properties = defaultdict(list)
        properties['num_nodes'].append(G.number_of_nodes())
        properties['num_edges'].append(G.number_of_edges())
        properties['avg_degree'].append(np.mean([d for n, d in G.degree()]))

        # Calculate importance scores for each edge type in the graph
        edge_type_importance = self._calculate_importance_scores(G)

        properties['edge_type_importance'] = edge_type_importance

        return properties

    def _calculate_importance_scores(self, G):
        edge_type_importance = {edge_type: 0 for edge_type in self.args.edge_types.values()}

        for u, v, data in G.edges(data=True):
            edge_type_importance[data['edge_attribute']] += data['weight']

        # Normalize importance
        total_edge_importance = sum(edge_type_importance.values())
        edge_type_importance = {self.edge_types_rev[k]: v / total_edge_importance for k, v in edge_type_importance.items()}

        return edge_type_importance