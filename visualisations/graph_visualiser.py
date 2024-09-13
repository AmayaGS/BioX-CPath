# graph_visualiser.py

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from PIL import Image
import ast


class GraphVisualiser:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.output_dir = os.path.join(args.directory, "graph_visualisations")
        os.makedirs(self.output_dir, exist_ok=True)


    def plot_global_edge_weight_distribution(self, fold_dir, all_graphs, fold):

        num_layers = len(next(iter(all_graphs.values())))
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 6 * num_layers), squeeze=False)

        layers = [[] for _ in range(num_layers)]
        for patient_graphs in all_graphs.values():
            for i, graph in enumerate(patient_graphs):
                layers[i].append(graph)

        for i, (layer_graphs) in enumerate(layers):
            global_edge_weights = {edge_type: [] for edge_type in self.args.edge_types.values()}
            for graphs in layer_graphs:
                for _, _, data in graphs.edges(data=True):
                    global_edge_weights[data['edge_attribute']].append(data['weight'])

            ax = axes[i, 0]
            for edge_type, weights in global_edge_weights.items():
                if weights:
                    edge_name = next(k for k, v in self.args.edge_types.items() if v == edge_type)
                    color = self.args.edge_colors[edge_name]
                    sns.histplot(weights, kde=True, stat="density", alpha=0.7,
                                 label=edge_name, color=color, ax=ax)

            ax.set_title(f"Layer {i + 1} Global Edge Weight Distribution by Edge Type")
            ax.set_xlabel("Edge Weight")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f"global_edge_weight_distribution_Fold_{fold}.png"))
        plt.close()

    def visualise_graphs(self, patient_id, patient_dir, all_data, fold):

        patient_graphs = all_data['graphs']
        metadata = all_data['metadata']

        num_layers = len(patient_graphs)
        edge_dist_fig, edge_dist_axes = plt.subplots(num_layers, 1, figsize=(12, 6 * num_layers), squeeze=False)

        for i, (G) in enumerate(patient_graphs):
            # Plot graph
            self._plot_graph(G, f"Layer {i + 1} Graph - Patient {patient_id}",
                             os.path.join(patient_dir, f"krag_graph_layer_{i + 1}_{patient_id}_Fold_{fold}.png"))

            # Visualise graph on WSI
            self._visualise_graph_on_wsi(metadata, G, i + 1, patient_id, patient_dir, fold)

            # Plot edge weight distribution
            self._plot_edge_weight_distribution(G, edge_dist_axes[i, 0], i + 1, patient_id)

        # Save edge weight distribution plot
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, f"edge_weight_distribution_{patient_id}_Fold_{fold}.png"))
        plt.close(edge_dist_fig)

    def _plot_graph(self, G, title, output_path):
        plt.figure(figsize=(14, 10))
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.spring_layout(G, k=0.5, iterations=50)


        # Plot nodes
        for stain_type, color in self.args.stain_colors.items():
            node_list = [node for node, data in G.nodes(data=True) if
                         data['stain_type'] == self.args.stain_types[stain_type]]
            if node_list:
                nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color,
                                       node_size=[G.nodes[node]['score'] for node in node_list], alpha=0.8)

        # Plot edges
        for edge_type, color in self.args.edge_colors.items():
            edge_list = [(u, v) for (u, v, data) in G.edges(data=True) if
                         data['edge_attribute'] == self.args.edge_types[edge_type]]
            if edge_list:
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color,
                                       width=[G[u][v]['weight'] for (u, v) in edge_list], alpha=0.6)

        # Create legends, including all stain types except default if not present
        stain_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=t,
                                            markerfacecolor=color, markersize=10)
                                for t, color in self.args.stain_colors.items()
                                if self.args.stain_types[t] != "NA"]
        edge_legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=t)
                                for t, color in self.args.edge_colors.items()]

        plt.legend(handles=stain_legend_elements + edge_legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(title, fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_edge_weight_distribution(self, G, ax, layer, patient_id):
        edge_weights = {edge_type: [] for edge_type in self.args.edge_types.values()}

        for _, _, data in G.edges(data=True):
            edge_weights[data['edge_attribute']].append(data['weight'])

        for edge_type, weights in edge_weights.items():
            if weights:
                edge_name = next(k for k, v in self.args.edge_types.items() if v == edge_type)
                color = self.args.edge_colors[edge_name]
                sns.histplot(weights, kde=True, stat="density", alpha=0.7,
                             label=edge_name, color=color, ax=ax)

        ax.set_title(f"Layer {layer} Edge Weight Distribution by Edge Type - Patient {patient_id}")
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Density")
        ax.legend()

    def _visualise_graph_on_wsi(self, metadata, G, layer, patient_id, output_dir, fold):

        fig, ax = plt.subplots(figsize=(30, 20))

        image_groups = self._group_patches_by_image(metadata)
        canvas, image_positions = self._create_wsi_canvas(image_groups)

        # graph_patches_metadata = {
        #     'filenames': [G.nodes[node]['filename'] for node in G.nodes()],
        #     'coordinates': [self._extract_coordinates(G.nodes[node]['filename']) for node in G.nodes()]
        # }

        pos = self._calculate_node_positions(G, metadata, image_positions, image_groups)

        ax.imshow(canvas)

        self._draw_nodes_on_wsi(G, pos, ax)
        self._draw_edges_on_wsi(G, pos, ax)
        self._add_wsi_labels(image_positions, canvas.shape[0], ax)

        plt.title(f"Layer {layer} Graph Overlay - Patient {patient_id}", fontsize=16)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"graph_overlay_layer_{layer}_{patient_id}_Fold_{fold}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _extract_coordinates(self, filename):
        import re
        coords = re.findall(r'row1=(\d+)_row2=(\d+)_col1=(\d+)_col2=(\d+)', filename)
        return list(map(int, coords[0])) if coords else None

    def _group_patches_by_image(self, patch_metadata):
        image_groups = {}
        for filename, coords in zip(patch_metadata['filenames'], patch_metadata['coordinates']):
            image_name = filename[0].split('_row1')[0]  # Extract image name without coordinates
            if image_name not in image_groups:
                image_groups[image_name] = []
            image_groups[image_name].append((filename[0], ast.literal_eval(coords)))

        return image_groups

    def _create_wsi_canvas(self, image_groups):
        # Calculate canvas size and image positions
        total_height = 0
        max_width = 0
        image_positions = {}
        image_heights = {}
        for image_name, patches in image_groups.items():
            coords = [patch[1] for patch in patches]
            height = max(coord[1] for coord in coords) - min(coord[0] for coord in coords)
            width = max(coord[3] for coord in coords)
            image_positions[image_name] = (0, total_height)  # All images start at x=0
            image_heights[image_name] = height
            total_height += height
            max_width = max(max_width, width)

        canvas = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        for image_name, patches in image_groups.items():
            _, image_offset_y = image_positions[image_name]
            for filename, coords in patches:
                y1, y2, x1, x2 = coords
                y1 -= min(coord[0] for _, coord in image_groups[image_name])  # Adjust y1 relative to image start
                y2 -= min(coord[0] for _, coord in image_groups[image_name])  # Adjust y2 relative to image start
                patch_path = os.path.join(self.args.path_to_patches, image_name, filename)
                if os.path.exists(patch_path):
                    patch_image = np.array(Image.open(patch_path))
                    if patch_image.shape[2] == 4:  # If RGBA, convert to RGB
                        patch_image = patch_image[:, :, :3]
                    canvas[y1 + image_offset_y:y2 + image_offset_y, x1:x2, :] = patch_image

        return canvas, image_positions

    def _calculate_node_positions(self, G, patch_metadata, image_positions, image_groups):
        pos = {}
        for node in G.nodes():
            filename = G.nodes[node]['filename']
            image_name = filename.split('_row1')[0]
            coords = next(coords for f, coords in zip(patch_metadata['filenames'], patch_metadata['coordinates']) if
                          f[0] == filename)  # problem here?
            coords = ast.literal_eval(coords)
            _, image_offset_y = image_positions[image_name]
            y_min = min(coord[0] for _, coord in image_groups[image_name])
            y = (coords[0] + coords[1]) / 2 - y_min + image_offset_y
            x = (coords[2] + coords[3]) / 2
            pos[node] = (x, y)

        return pos

    def _draw_nodes_on_wsi(self, G, pos, ax):
        for stain_type, color in self.args.stain_colors.items():
            node_list = [node for node, data in G.nodes(data=True) if data['stain_type'] == self.args.stain_types[stain_type]]
            if node_list:
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list, node_color=color,
                                       node_size=[G.nodes[node]['score'] for node in node_list],
                                       alpha=0.6)

    def _draw_edges_on_wsi(self, G, pos, ax):
        for edge_type, color in self.args.edge_colors.items():
            edge_list = [(u, v) for (u, v, data) in G.edges(data=True) if data['edge_attribute'] == self.args.edge_types[edge_type]]
            if edge_list:
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list, edge_color=color,
                                       width=[G[u][v]['weight'] for (u, v) in edge_list],
                                       alpha=0.6)

    def _add_wsi_labels(self, image_positions, total_height, ax):
        for image_name, (_, offset_y) in image_positions.items():
            ax.text(0.02, 1 - (offset_y + 10) / total_height, image_name, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))