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

    def visualise_graphs(self, all_patient_data, fold_dir, fold):

        for patient in all_patient_data.keys():

            patient_dir = os.path.join(fold_dir, patient)
            os.makedirs(patient_dir, exist_ok=True)
            patient_graphs = all_patient_data[patient]['graphs']
            metadata = all_patient_data[patient]['metadata']

            for i, (G) in enumerate(patient_graphs):

                # Visualise graph on WSI
                self._visualise_graph_on_wsi(metadata, G, i + 1, patient, patient_dir, fold)

                # Plot graph
                # self._plot_graph(G, f"Layer {i + 1} Graph - Patient {patient}",
                #                  os.path.join(patient_dir, f"krag_graph_layer_{i + 1}_{patient}_Fold_{fold}.png"))

    def _plot_graph(self, G, title, output_path, sampling_rate=0.3, layout_seed=42):
        """
        Plot graph with stochastic edge sampling and optimized layout computation.

        Args:
            G (networkx.Graph): Input graph
            title (str): Plot title
            output_path (str): Path to save visualization
            sampling_rate (float): Edge sampling rate [0,1]
            layout_seed (int): Random seed for reproducible layout
        """
        plt.figure(figsize=(14, 10))

        # Remove self-loops and isolates for cleaner visualization
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))

        # Optimize layout computation using ForceAtlas2
        np.random.seed(layout_seed)
        pos = nx.spring_layout(
            G,
            k=0.8,  # Optimal repulsion strength for biological networks
            iterations=50,  # Increased for better convergence
            weight='weight',  # Considers edge weights in layout
            scale=2.0  # Scaling factor for better node separation
        )

        # Plot nodes with attention to perceptual optimization
        for stain_type, color in self.args.stain_colors.items():
            node_list = [node for node, data in G.nodes(data=True)
                         if data['stain_type'] == self.args.stain_types[stain_type]]
            if node_list:
                # Compute node sizes with logarithmic scaling for better visual distribution
                node_sizes = [np.log1p(G.nodes[node]['score']) * 10 for node in node_list]

                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=node_list,
                    node_color=color,
                    node_size=node_sizes,
                    alpha=0.8,
                    edgecolors='white',  # Add edge contrast
                    linewidths=0.5
                )

        # Plot edges with stochastic sampling
        for edge_type, color in self.args.edge_colors.items():
            edge_list = [(u, v) for (u, v, data) in G.edges(data=True)
                         if data['edge_attribute'] == self.args.edge_types[edge_type]]

            if edge_list:
                # # Implement edge sampling with weight-based importance
                # edge_weights = np.array([G[u][v]['weight'] for (u, v) in edge_list])
                # sampling_probs = edge_weights / edge_weights.sum()

                # n_samples = int(len(edge_list) * sampling_rate)
                # sampled_indices = np.random.choice(
                #     len(edge_list),
                #     size=n_samples,
                #     replace=False,
                #     p=sampling_probs
                # )

                # Weight-based thresholding
                weight_threshold = np.percentile([G[u][v]['weight'] for (u, v) in edge_list], 70)
                filtered_edges = [(u, v) for (u, v) in edge_list if G[u][v]['weight'] >= weight_threshold]

                # Adjust sampling rate based on filtered subset
                effective_sampling_rate = min(sampling_rate * len(edge_list) / len(filtered_edges), 1.0)

                sampled_indices = np.random.choice(
                    len(filtered_edges),
                    size=int(len(filtered_edges) * effective_sampling_rate),
                    replace=False
                )

                sampled_edges = [edge_list[i] for i in sampled_indices]

                # Draw edges with width scaling
                edge_widths = [np.log1p(G[u][v]['weight']) * 0.5 for (u, v) in sampled_edges]

                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=sampled_edges,
                    edge_color=color,
                    width=edge_widths,
                    alpha=0.6,
                    style='solid'
                )

        # Enhanced legend creation with hierarchical organization
        stain_legend_elements = [
            plt.Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=t,
                       markerfacecolor=color,
                       markersize=10,
                       markeredgecolor='white',
                       markeredgewidth=0.5)
            for t, color in self.args.stain_colors.items()
            if self.args.stain_types[t] != "NA"
        ]

        edge_legend_elements = [
            plt.Line2D([0], [0],
                       color=color,
                       lw=2,
                       label=t,
                       alpha=0.8)
            for t, color in self.args.edge_colors.items()
        ]

        # Create two-column legend with optimized placement
        plt.legend(
            handles=stain_legend_elements + edge_legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=1
        )

        plt.title(title, fontsize=20, pad=20)
        plt.axis('off')

        # Optimize figure layout and save with high quality
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            transparent=False
        )
        plt.close()

    def _visualise_graph_on_wsi(self, metadata, G, layer, patient_id, output_dir, fold):

        self.logger.info(f"Plotting graph on WSIs for patient {patient_id} - Layer {layer}")

        fig, ax = plt.subplots(figsize=(30, 20))

        image_groups = self._group_patches_by_image(metadata)
        canvas, image_positions = self._create_wsi_canvas(image_groups)
        pos = self._calculate_node_positions(G, metadata, image_positions, image_groups)

        ax.imshow(canvas)
        ax.set_xlim(0, canvas.shape[1])
        ax.set_ylim(canvas.shape[0], 0)

        self._draw_nodes_on_wsi(G, pos, ax)
        self._draw_edges_on_wsi(G, pos, ax)
        self._add_wsi_labels(image_positions, canvas.shape[0], ax)

        plt.title(f"Layer {layer} Graph Overlay - Patient {patient_id}", fontsize=16)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"graph_overlay_layer_{layer}_{patient_id}_Fold_{fold}.png")
                    , dpi=300, bbox_inches='tight')
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
                          f[0] == filename)  # problem  for wsi labels?
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
                                       node_size=[(G.nodes[node]['score']) for node in node_list],
                                       alpha=0.6)

                # nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list, node_color=color,
                #                        node_size=[np.log1p(G.nodes[node]['score']) * 10 for node in node_list],
                #                        alpha=0.6)

    def _draw_edges_on_wsi(self, G, pos, ax, sampling_rate=0.3):
        for edge_type, color in self.args.edge_colors.items():

            edge_list = [(u, v) for (u, v, data) in G.edges(data=True)
                         if data['edge_attribute'] == self.args.edge_types[edge_type]
                         and u != v]

            if edge_list:
                edge_weights = np.array([G[u][v]['weight'] for (u, v) in edge_list])
                sampling_probs = edge_weights / edge_weights.sum()  # Normalize weights

                # Sample edges using numpy's random choice with probability weights
                n_samples = int(len(edge_list) * sampling_rate)
                sampled_indices = np.random.choice(
                    len(edge_list),
                    size=n_samples,
                    replace=False,
                    p=sampling_probs
                )

                # Get sampled edges
                sampled_edges = [edge_list[i] for i in sampled_indices]

                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sampled_edges, edge_color=color,
                                       width=[(G[u][v]['weight']) * 1.2 for (u, v) in sampled_edges],
                                       alpha=0.6)
                #
                # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sampled_edges, edge_color=color,
                #                        width=[np.log1p(G[u][v]['weight']) * 0.5 for (u, v) in sampled_edges],
                #                        alpha=0.6)
                #

    def _add_wsi_labels(self, image_positions, total_height, ax):
        for image_name, (_, offset_y) in image_positions.items():
            normalized_y = (offset_y) / total_height
            ax.text(0.02, 1 - normalized_y, image_name, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

