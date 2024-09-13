import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

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
            self._visualise_top_k_nodes(patient_id, patient_dir, label, layer, properties)
        metrics[patient_id].append(layers)

        return metrics

    def calculate_graph_properties(self, G, patient_id, label, layer, fold_dir, fold):
        properties = defaultdict(list)
        properties['num_nodes'].append(G.number_of_nodes())
        properties['num_edges'].append(G.number_of_edges())
        properties['avg_degree'].append(np.mean([d for n, d in G.degree()]))

        # Node centrality measures
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=10000)
        node_attention_score = nx.get_node_attributes(G,'score')

        properties['degree_centrality'].append(degree_centrality)
        properties['closeness_centrality'].append(closeness_centrality)
        properties['betweenness_centrality'].append(betweenness_centrality)
        properties['eigenvector_centrality'].append(eigenvector_centrality)
        properties['node_attention_score'].append(node_attention_score)

        # Edge importance
        edge_betweenness = nx.edge_betweenness_centrality(G)

        properties['edge_betweenness'].append(edge_betweenness)

        # Calculate importance scores for each stain type and edge type in the graph
        stain_importance, edge_type_importance = self._calculate_importance_scores(G)

        properties['stain_importance'] = stain_importance
        properties['edge_type_importance'] = edge_type_importance

        stain_properties = self._aggregate_node_properties(G, degree_centrality, closeness_centrality,
                                   betweenness_centrality, eigenvector_centrality)

        properties['stain_properties'] = stain_properties

        # Calculate average properties for each stain type
        for stain_type, measures in stain_properties.items():
            for measure, values in measures.items():
                properties[f'{stain_type}_{measure}_avg'].append(np.mean(values))
                properties[f'{stain_type}_{measure}_max'].append(np.max(values))

        measures = ['degree_centrality', 'closeness_centrality', 'betweenness_centrality',
                    'eigenvector_centrality', 'node_attention_score']
        # Top 5 most central nodes for each measure
        for measure in measures:
            centrality_dict = properties[measure][0]
            top_nodes = sorted(centrality_dict, key=centrality_dict.get, reverse=True)[:5]
            properties[f'top_5_{measure}'].append([
                {   'node': node,
                    'filename': G.nodes[node]['filename'],
                    'stain_type': self.stain_types_rev[G.nodes[node]['stain_type']],
                    'score': centrality_dict[node]
                }
                for node in top_nodes
            ])

        # Top 5 most important edges
        top_edges = sorted(edge_betweenness, key=edge_betweenness.get, reverse=True)[:5]
        properties['top_5_edge_betweenness'].append(
            [(edge, G.edges[edge]['edge_attribute'], edge_betweenness[edge]) for edge in top_edges]
        )

        return properties

    def _calculate_importance_scores(self, G):
        stain_importance = {stain: 0 for stain in self.args.stain_types.values()}
        edge_type_importance = {edge_type: 0 for edge_type in self.args.edge_types.values()}

        for node, data in G.nodes(data=True):
            stain_importance[data['stain_type']] += data['score']

        for u, v, data in G.edges(data=True):
            edge_type_importance[data['edge_attribute']] += data['weight']

        # Normalize importances
        total_stain_importance = sum(stain_importance.values())
        total_edge_importance = sum(edge_type_importance.values())

        stain_importance = {self.stain_types_rev[k]: v / total_stain_importance for k, v in stain_importance.items()}
        edge_type_importance = {self.edge_types_rev[k]: v / total_edge_importance for k, v in edge_type_importance.items()}

        return stain_importance, edge_type_importance

    def _aggregate_node_properties(self, G, degree_centrality, closeness_centrality,
                                   betweenness_centrality, eigenvector_centrality):
        stains =  list(self.args.stain_types.keys())
        stain_properties = {stain: defaultdict(list) for stain in stains}
        for node, data in G.nodes(data=True):
            stain_type = self.stain_types_rev[data['stain_type']]
            stain_properties[stain_type]['degree_centrality'].append(degree_centrality[node])
            stain_properties[stain_type]['closeness_centrality'].append(closeness_centrality[node])
            stain_properties[stain_type]['betweenness_centrality'].append(betweenness_centrality[node])
            stain_properties[stain_type]['eigenvector_centrality'].append(eigenvector_centrality[node])
            stain_properties[stain_type]['node_attention_score'].append(data['score'])
        stain_properties = {stain: properties for stain, properties in stain_properties.items() if properties}

        return stain_properties


    def _visualise_top_k_nodes(self, patient_id, patient_dir, label, layer, properties):
        measures = ['degree_centrality', 'closeness_centrality', 'betweenness_centrality',
                    'eigenvector_centrality', 'node_attention_score']

        fig = plt.figure(figsize=(20, 22))
        gs = gridspec.GridSpec(6, len(measures), height_ratios=[0.25, 1, 1, 1, 1, 1])

        plt.suptitle(f"Top 5 nodes - Patient ID: {patient_id}, Layer {layer} Label: {self.args.label_dict[str(label)]}",
                     fontsize=25, y=0.98)

        # Add column titles
        for i, measure in enumerate(measures):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, measure.replace('_', ' ').title(),
                    fontsize=15, ha='center', va='center')
            ax.axis('off')

        for i, measure in enumerate(measures):
            top_nodes = properties[f'top_5_{measure}']

            scores = [node['score'] for nodes in top_nodes for node in nodes]
            min_score, max_score = min(scores), max(scores)
            norm = plt.Normalize(vmin=min_score, vmax=max_score)

            for j, nodes in enumerate(top_nodes):
                for v, node in enumerate(nodes):
                    ax = fig.add_subplot(gs[v + 1, i])
                    ax.axis('off')
                    filename = node['filename'].split("_row1")[0]
                    path = os.path.join(self.args.path_to_patches, filename, node['filename'])
                    img = Image.open(path)
                    ax.imshow(img)
                    ax.set_title(f"{node['stain_type']}\n{node['score']:.2f}", fontsize=12)

                    score = node['score']
                    color = plt.cm.viridis(norm(score))
                    rect = plt.Rectangle((0, 0), img.width, img.height, fill=False,
                                         edgecolor=color, linewidth=2 + 3 * norm(score))

                    ax.add_patch(rect)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
        plt.savefig(f"{patient_dir}/{patient_id}_Layer_{layer}_top_centrality_measures.png", dpi=300,
                    bbox_inches='tight')
        plt.close()











        # measures = ['degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'node_score']
        #
        # for measure in measures:
        #     top_nodes = properties[f'top_5_{measure}']
        #     for i, nodes in enumerate(top_nodes):
        #         fig, ax = plt.subplots(1, 5, figsize=(10, 2))
        #         for j, node in enumerate(nodes):
        #             filename = node['filename'].split("_row1")[0]
        #             path = os.path.join(self.args.path_to_patches, filename, node['filename'])
        #             img = Image.open(path)
        #             ax[j].imshow(img)
        #             ax[j].set_title(f"{node['stain_type']}\n{node['score']:.2f}")
        #             ax[j].axis('off')
        #         plt.tight_layout()
        #         plt.savefig(f"{patient_dir}/top_{measure}_{i}.png")
        #         plt.close()

#
# Schema of nested dictionary structure
# {Patient ID: {Layer # :{
#     'num_nodes': [int],
#     'num_edges': [int],
#     'avg_degree': [float],
#     'degree_centrality': [{node_id: float, ...}],
#     'closeness_centrality': [{node_id: float, ...}],
#     'betweenness_centrality': [{node_id: float, ...}],
#     'eigenvector_centrality': [{node_id: float, ...}],
#     'edge_betweenness': [{(node1, node2): float, ...}],
#     'stain_importance': {
#         'NA': float,
#         'H&E': float,
#         'CD68': float,
#         'CD138': float,
#         'CD20': float
#     },
#     'edge_type_importance': {
#         'RAG': float,
#         'KNN': float,
#         'BOTH': float
#     },
#     'stain_properties': {
#         'CD138': {
#             'degree_centrality': [float, ...],
#             'closeness_centrality': [float, ...],
#             'betweenness_centrality': [float, ...],
#             'eigenvector_centrality': [float, ...],
#             'node_score': [float, ...]
#         }
#     },
#     'CD138_degree_centrality_avg': [float],
#     'CD138_degree_centrality_max': [float],
#     'CD138_closeness_centrality_avg': [float],
#     'CD138_closeness_centrality_max': [float],
#     'CD138_betweenness_centrality_avg': [float],
#     'CD138_betweenness_centrality_max': [float],
#     'CD138_eigenvector_centrality_avg': [float],
#     'CD138_eigenvector_centrality_max': [float],
#     'CD138_node_score_avg': [float],
#     'CD138_node_score_max': [float],
#     'top_10_degree_centrality': [[
#         {
#             'node': int,
#             'filename': str,
#             'stain_type': str,
#             'score': float
#         },
#         ...
#     ]],
#     'top_10_closeness_centrality': [[...]], # Same structure as top_10_degree_centrality
#     'top_10_betweenness_centrality': [[...]], # Same structure as top_10_degree_centrality
#     'top_10_eigenvector_centrality': [[...]], # Same structure as top_10_degree_centrality
#     'top_5_edge_betweenness': [[
#         ((node1, node2), edge_type, score),
#         ...
#     ]]
# }}
#.
#.}
# #
# # Reverse the dictionaries for easy lookup
# stain_types_rev = {v: k for k, v in self.args.stain_types.items()}
# edge_types_rev = {v: k for k, v in self.args.edge_types.items()}
#
# # Save importance scores
# os.makedirs(fold_dir, exist_ok=True)
# with open(f"{fold_dir}/importance_scores_layer_{layer}_{patient_id}.txt", 'w') as f:
#     f.write("Stain Type Importance:\n")
#     for stain, importance in stain_importance.items():
#         stain_name = stain_types_rev.get(stain, f'Unknown ({stain})')
#         f.write(f"{stain_name}: {importance:.4f}\n")
#         properties[f'stain_importance_{stain_name}'].append(importance)
#
#     f.write("\nEdge Type Importance:\n")
#     for edge_type, importance in edge_type_importance.items():
#         edge_type_name = edge_types_rev.get(edge_type, f'Unknown ({edge_type})')
#         f.write(f"{edge_type_name}: {importance:.4f}\n")
#         properties[f'edge_type_importance_{edge_type_name}'].append(importance)