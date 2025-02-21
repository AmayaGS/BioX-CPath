import networkx as nx
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


class StainRelationship:
    def __init__(self, args):
        self.stain_names = {v: k for k, v in args.stain_types.items()}
        self.args = args

    def plot_stain_relationships(self, all_patient_data, fold_dir, subset_type):

        multi_stain_patients = self._get_multi_stain_patients(all_patient_data)

        if subset_type == 'multistain':
            all_patient_data = multi_stain_patients

        all_patient_layer_analysis = self._load_layer_analysis(all_patient_data, fold_dir, subset_type)

        for patient_id in all_patient_data.keys():
            patient_data = all_patient_data[patient_id]
            layer_analysis = all_patient_layer_analysis[patient_id]

            self.analyze_patient_relationships(patient_data, layer_analysis, patient_id, fold_dir)

        # Analyze relationships across the entire dataset
        self.analyze_dataset_relationships(all_patient_data, all_patient_layer_analysis, fold_dir, subset_type)

    def analyze_patient_relationships(self, patient_data, layer_analyses, patient_id, fold_dir):

        # add functions for patient level analysis here
        for layer_idx in layer_analyses:

            # Create layer-specific visualizations
            self._plot_layer_visualizations(layer_analyses[layer_idx],
                                            layer_idx,
                                            patient_id,
                                            fold_dir)


    def analyze_dataset_relationships(self, all_patient_data, layer_analyses, fold_dir, subset_type):
        """
        Analyze stain relationships across all patients and create label comparisons
        """
        all_patient_stats = {}

        # Collect data for each label
        for patient_id, patient_data in all_patient_data.items():
            stats = pd.concat(layer_analyses[patient_id].values())
            all_patient_stats[patient_id] = stats

        dataset_stats = pd.concat(all_patient_stats)
        avg_correlations = self._calculate_dataset_averages(dataset_stats)

        self._create_dataset_visualizations(dataset_stats, avg_correlations, fold_dir, subset_type)

    def _get_multi_stain_patients(self, all_metrics):

        multi_stain_patients = {}

        for patient_id, patient_data in all_metrics.items():
            # Get unique stains across all layers
            unique_stains = set()
            for stain_dict in patient_data['stain_attention']:  # stain_attentions
                # Only count non-empty stain attentions
                current_stains = {stain for stain, value in stain_dict.items()
                                  if value > 0}  # Filter out zero-value stains
                unique_stains.update(current_stains)

            # If patient has more than one stain type (excluding any NA stains)
            active_stains = {stain for stain in unique_stains
                             if str(stain) != 'NA' and str(stain) != '-1'}

            if len(active_stains) > 1:
                multi_stain_patients[patient_id] = patient_data

        return multi_stain_patients

    def _load_layer_analysis(self, all_patient_data, fold_dir, subset_type):

        layer_analysis_path = Path(fold_dir) / f'layer_analysis_{subset_type}.pkl'
        if os.path.exists(layer_analysis_path):
            with open(layer_analysis_path, 'rb') as f:
                all_patient_layer_analysis = pickle.load(f)
        else:
            all_patient_layer_analysis = {}
            for patient_id, patient_data in all_patient_data.items():
                layer_analyses = {}
                # Analyze each layer for this patient
                for layer_idx, (layer_data, G) in enumerate(zip(patient_data['layer_data'][patient_id],
                                                                patient_data['graphs'])):
                    # Calculate statistics for this layer
                    layer_stats = self._calculate_layer_statistics(patient_data, G, layer_data, layer_idx)
                    layer_analyses[layer_idx] = pd.DataFrame(layer_stats)

                all_patient_layer_analysis[patient_id] = layer_analyses

            with open(layer_analysis_path, 'wb') as f:
                pickle.dump(all_patient_layer_analysis, f)

        return all_patient_layer_analysis

    def _calculate_layer_statistics(self, patient_data, G, layer_data, layer_idx):
        """
        Calculate statistics for a single layer
        """
        stats = []
        stain_pair_weights = defaultdict(list)
        stain_pair_edge_types = defaultdict(list)

        # Collect weights and edge types for each stain pair
        for u, v, edge_data in G.edges(data=True):

            # Get stain types
            source_stain = G.nodes[u]['stain_type']
            target_stain = G.nodes[v]['stain_type']

            # Get edge weight
            weight = edge_data['weight']

            # Get edge type if available
            edge_type = (edge_data['edge_attribute']
                         if hasattr(edge_data, 'edge_attribute') else None)

            # Store data
            stain_pair = tuple(sorted([source_stain, target_stain]))
            stain_pair_weights[stain_pair].append(weight)
            if edge_type is not None:
                stain_pair_edge_types[stain_pair].append(edge_type)

        # Calculate statistics for each stain pair
        for stain_pair, weights in stain_pair_weights.items():
            weights_array = np.array(weights)
            edge_types = stain_pair_edge_types.get(stain_pair, [])

            stat = {
                'layer': layer_idx,
                'label': patient_data['actual_label'],
                'predicted_label': patient_data['predicted_label'],
                'stain_1': self.stain_names[stain_pair[0]],
                'stain_2': self.stain_names[stain_pair[1]],
                'mean_weight': np.mean(weights_array),
                'median_weight': np.median(weights_array),
                'std_weight': np.std(weights_array),
                'max_weight': np.max(weights_array),
                'min_weight': np.min(weights_array),
                'connection_count': len(weights),
                'spatial_connections': sum(1 for et in edge_types if et == 0),
                'feature_connections': sum(1 for et in edge_types if et == 1)
            }
            stats.append(stat)

        return stats


    # def _calculate_layer_statistics(self, patient_data, G, layer_data, layer_idx):
    #     """
    #     Calculate statistics for a single layer
    #     """
    #     stats = []
    #     stain_pair_weights = defaultdict(list)
    #     stain_pair_edge_types = defaultdict(list)
    #
    #     # Collect weights and edge types for each stain pair
    #     for u, v, edge_data in G.edges(data=True):
    #         # Get original node indices
    #         original_u = layer_data.node_mapping[u]
    #         original_v = layer_data.node_mapping[v]
    #
    #         # Get stain types
    #         source_stain = layer_data.node_attr[original_u].item()
    #         target_stain = layer_data.node_attr[original_v].item()
    #
    #         # Get edge weight
    #         original_edge = str([original_u, original_v])
    #         original_edge_index = layer_data.edge_mapping[original_edge]
    #         attention_idx = layer_data.edge_idx_mapping[original_edge_index]
    #         weight = edge_data['weight']
    #
    #         # Get edge type if available
    #         edge_type = (layer_data.original_edge_attr[original_edge_index].item()
    #                      if hasattr(layer_data, 'original_edge_attr') else None)
    #
    #         # Store data
    #         stain_pair = tuple(sorted([source_stain, target_stain]))
    #         stain_pair_weights[stain_pair].append(weight)
    #         if edge_type is not None:
    #             stain_pair_edge_types[stain_pair].append(edge_type)
    #
    #     # Calculate statistics for each stain pair
    #     for stain_pair, weights in stain_pair_weights.items():
    #         weights_array = np.array(weights)
    #         edge_types = stain_pair_edge_types.get(stain_pair, [])
    #
    #         stat = {
    #             'layer': layer_idx,
    #             'label': patient_data['actual_label'],
    #             'predicted_label': patient_data['predicted_label'],
    #             'stain_1': self.stain_names[stain_pair[0]],
    #             'stain_2': self.stain_names[stain_pair[1]],
    #             'mean_weight': np.mean(weights_array),
    #             'median_weight': np.median(weights_array),
    #             'std_weight': np.std(weights_array),
    #             'max_weight': np.max(weights_array),
    #             'min_weight': np.min(weights_array),
    #             'connection_count': len(weights),
    #             'spatial_connections': sum(1 for et in edge_types if et == 0),
    #             'feature_connections': sum(1 for et in edge_types if et == 1)
    #         }
    #         stats.append(stat)
    #
    #     return stats


    def _plot_layer_visualizations(self, stats_df, layer_idx, patient_id, fold_dir):
        """
        Create visualizations for a single layer
        """
        output_dir = Path(fold_dir) / str(patient_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create heatmap of attention weights
        plt.figure(figsize=(12, 10))
        pivot_df = stats_df.pivot(index='stain_1', columns='stain_2', values='mean_weight')

        pivot_df = pivot_df.fillna(pivot_df.T)
        symmetric_matrix = (pivot_df + pivot_df.T) / 2
        sns.heatmap(symmetric_matrix, annot=True, fmt='.2f', cmap='viridis')

        plt.title(f'Layer {layer_idx} - Mean Attention Weights')
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_heatmap.png')
        plt.close()


    def _calculate_dataset_averages(self, dataset_stats):
        """
        Calculate average statistics across the dataset
        """
        avg_correlations = dataset_stats.groupby(['stain_1', 'stain_2', 'layer', 'label']).agg({
            'mean_weight': ['mean', 'std'],
            'connection_count': 'mean',
            'spatial_connections': 'mean',
            'feature_connections': 'mean'
        }).reset_index()

        # Flatten column names
        avg_correlations.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                    for col in avg_correlations.columns]

        return avg_correlations

    def _create_dataset_visualizations(self, dataset_stats, avg_correlations, fold_dir, subset_type):

        # Add all dataset level visualizations here
        # 1. Average attention heatmap across all patients and layers
        self._plot_dataset_heatmap(avg_correlations, fold_dir,('mean_weight_mean', 'mean'), subset_type)

        # 4. Distribution of attention weights
        self._plot_stain_distributions(dataset_stats, fold_dir, subset_type)


    def _plot_stain_distributions(self, data_df, fold_dir, subset_type):

        all_stains = sorted(set(data_df['stain_1'].unique()) | set(data_df['stain_2'].unique()))

        # Create a consistent color palette
        n_colors = len(all_stains)
        color_palette = sns.color_palette("tab10", n_colors)
        stain_colors = dict(zip(all_stains, color_palette))

        # Create symmetric pairs by duplicating in reverse order
        symmetric_rows = []
        for _, row in data_df.iterrows():
            symmetric_rows.append(row)
            # Add reversed pair
            new_row = row.copy()
            new_row['stain_1'], new_row['stain_2'] = row['stain_2'], row['stain_1']
            symmetric_rows.append(new_row)

        data_df = pd.DataFrame(symmetric_rows)

        y_min = 0
        y_max = data_df['mean_weight'].max()

        for label in sorted(data_df['label'].unique()):
            # Filter data for current label
            df_label = data_df[data_df['label'] == label].copy()

            # 2. Box plot
            plt.figure(figsize=(15, 8))
            box = sns.boxplot(data=df_label,
                              x='stain_1',
                              y='mean_weight',
                              hue='stain_2',
                              showfliers=False,
                              order=all_stains,
                              hue_order=all_stains,
                              palette=stain_colors)
            plt.ylim(y_min, y_max)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Attention Weight')
            plt.title(f'Distribution of Attention Weights - {self.args.label_dict[str(label)]} ({subset_type})')

            handles = box.legend_.legend_handles
            plt.legend(handles,
                       all_stains,
                       title='Stain 2',
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, f'label_{label}_attention_box_{subset_type}.png'),
                        bbox_inches='tight')
            plt.close()

    # def _plot_stain_distributions(self, data_df, fold_dir, subset_type):
    #
    #     all_stains = sorted(set(data_df['stain_1'].unique()) | set(data_df['stain_2'].unique()))
    #
    #     # Create a consistent color palette
    #     n_colors = len(all_stains)
    #     color_palette = sns.color_palette("tab10", n_colors)
    #     stain_colors = dict(zip(all_stains, color_palette))
    #
    #     # Ensure stain pairs are always in consistent order
    #     def order_stain_pair(row):
    #         stain1, stain2 = row['stain_1'], row['stain_2']
    #         if stain1 > stain2:
    #             row['stain_1'], row['stain_2'] = stain2, stain1
    #         return row
    #
    #     data_df = data_df.apply(order_stain_pair, axis=1)
    #
    #     y_min = 0
    #     y_max = data_df['mean_weight'].max()
    #
    #     for label in sorted(data_df['label'].unique()):
    #         # Filter data for current label
    #         df_label = data_df[data_df['label'] == label].copy()
    #
    #         # 2. Box plot
    #         plt.figure(figsize=(15, 8))
    #         box = sns.boxplot(data=df_label,
    #                     x='stain_1',
    #                     y='mean_weight',
    #                     hue='stain_2',
    #                     showfliers=False,
    #                     order=all_stains,
    #                     hue_order=all_stains,
    #                     palette=stain_colors)
    #         plt.ylim(y_min, y_max)
    #         plt.xticks(rotation=45, ha='right')
    #         plt.ylabel('Attention Weight')
    #         plt.title(f'Distribution of Attention Weights - {self.args.label_dict[str(label)]} ({subset_type})')
    #
    #         handles = box.legend_.legend_handles
    #         plt.legend(handles,
    #                    all_stains,
    #                    title='Stain 2',
    #                    bbox_to_anchor=(1.05, 1),
    #                    loc='upper left')
    #
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(fold_dir, f'label_{label}_attention_box_{subset_type}.png'),
    #                     bbox_inches='tight')
    #         plt.close()
    #
    #     self._export_statistics(data_df, fold_dir, subset_type)


    def _export_statistics(self, df, fold_dir, subset_type):

        stats_data = []

        # Split data by label
        df_label0 = df[df['label'] == 0]
        df_label1 = df[df['label'] == 1]

        # Calculate global statistics
        all_weights = df['mean_weight'].values
        global_mean = np.mean(all_weights)
        global_std = np.std(all_weights)

        # Get unique stains in consistent order
        all_stains = sorted(set(df['stain_1'].unique()) | set(df['stain_2'].unique()))

        for stain1 in all_stains:
            for stain2 in all_stains:
                # Get data for both labels
                subset0 = df_label0[(df_label0['stain_1'] == stain1) &
                                    (df_label0['stain_2'] == stain2)]
                subset1 = df_label1[(df_label1['stain_1'] == stain1) &
                                    (df_label1['stain_2'] == stain2)]

                if not subset0.empty and not subset1.empty:
                    mean0 = subset0['mean_weight'].mean()
                    mean1 = subset1['mean_weight'].mean()
                    std0 = subset0['std_weight'].mean()  # Using provided std
                    std1 = subset1['std_weight'].mean()  # Using provided std

                    # Calculate changes
                    abs_diff = abs(mean1 - mean0)
                    percent_change = ((mean1 - mean0) / mean0) * 100
                    relative_change = abs_diff / global_mean * 100
                    normalized_diff = abs_diff / global_std

                    # Categorize change magnitude
                    def categorize_attention_change(rel_change, norm_diff):
                        if rel_change > 15 or norm_diff > 0.5:
                            return "Major"
                        elif rel_change > 5 or norm_diff > 0.2:
                            return "Moderate"
                        elif rel_change > 2 or norm_diff > 0.1:
                            return "Minor"
                        else:
                            return "Minimal"

                    stats_data.append({
                        'Stain_1': stain1,
                        'Stain_2': stain2,
                        'Label0_Mean': mean0,
                        'Label1_Mean': mean1,
                        'Label0_Std': std0,
                        'Label1_Std': std1,
                        'Absolute_Difference': abs_diff,
                        'Percent_Change': percent_change,
                        'Relative_Change': relative_change,
                        'Normalized_Difference': normalized_diff,
                        'Change_Category': categorize_attention_change(relative_change, normalized_diff),
                        'Direction': 'Increased in Label 1' if mean1 > mean0 else 'Decreased in Label 1',
                        'Sample_Size_Label0': len(subset0),
                        'Sample_Size_Label1': len(subset1)
                    })

        # Create DataFrame and sort by absolute difference
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values('Percent_Change', key=abs, ascending=False)

        # Export to CSV
        output_path = os.path.join(fold_dir, f'attention_pattern_changes_{subset_type}.csv')
        stats_df.to_csv(output_path, index=False)

        # Create summary focusing on attention patterns
        summary_path = os.path.join(fold_dir, f'attention_pattern_summary_{subset_type}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Changes in Attention Patterns ({subset_type}):\n\n")

            # Major changes
            major = stats_df[stats_df['Change_Category'] == 'Major']
            f.write("Major Changes in Attention:\n")
            for _, row in major.iterrows():
                f.write(f"• {row['Stain_1']}-{row['Stain_2']}: {row['Percent_Change']:.1f}% {row['Direction']}\n")

            # Moderate changes
            moderate = stats_df[stats_df['Change_Category'] == 'Moderate']
            f.write("\nModerate Changes in Attention:\n")
            for _, row in moderate.iterrows():
                f.write(f"• {row['Stain_1']}-{row['Stain_2']}: {row['Percent_Change']:.1f}% {row['Direction']}\n")

            # Most stable interactions
            stable = stats_df[stats_df['Change_Category'] == 'Minimal']
            f.write("\nMost Stable Attention Patterns:\n")
            for _, row in stable.iterrows():
                f.write(f"• {row['Stain_1']}-{row['Stain_2']}: {row['Percent_Change']:.1f}% change\n")

            # Add summary statistics
            f.write("\nSummary Statistics:\n")
            f.write(f"Global Mean: {global_mean:.3f}\n")
            f.write(f"Global Standard Deviation: {global_std:.3f}\n")
            f.write(f"Total Stain Pairs Analyzed: {len(stats_df)}\n")
            f.write(f"Number of Major Changes: {len(major)}\n")
            f.write(f"Number of Moderate Changes: {len(moderate)}\n")
            f.write(f"Number of Stable Patterns: {len(stable)}\n")

        return stats_df


    def _plot_dataset_heatmap(self, stats_df, fold_dir, value_col, subset_type):
        """
        Create and save a heatmap visualization
        """
        # Create correlation matrix
        labels = sorted(stats_df['label'].unique())
        unique_stains = sorted(set(stats_df['stain_1'].unique()) | set(stats_df['stain_2'].unique()))
        n_stains = len(unique_stains)

        # Get average values for each stain pair
        column_name = f"{value_col[0]}"  # e.g., 'mean_weight_mean'
        label_matrices = {}

        for label in labels:
            label_df = stats_df[stats_df['label'] == label]
            correlation_matrix = np.zeros((n_stains, n_stains))
            avg_data = []
            plotted_pairs = set()

            for _, row in label_df.iterrows():
                stain1, stain2 = row['stain_1'], row['stain_2']

                pair = frozenset([stain1, stain2])
                if pair not in plotted_pairs:
                    # Get data for both directions
                    pair_data = label_df[
                        ((label_df['stain_1'] == stain1) & (label_df['stain_2'] == stain2)) |
                        ((label_df['stain_1'] == stain2) & (label_df['stain_2'] == stain1))
                        ]
                    avg_value = pair_data[column_name].mean()
                    avg_data.append({
                        'stain_1': stain1,
                        'stain_2': stain2,
                        column_name: avg_value
                    })
                    plotted_pairs.add(pair)

            avg_data = pd.DataFrame(avg_data)

            # Fill correlation matrix symmetrically
            for _, row in avg_data.iterrows():
                i = unique_stains.index(row['stain_1'])
                j = unique_stains.index(row['stain_2'])
                value = row[column_name]
                correlation_matrix[i, j] = value
                correlation_matrix[j, i] = value  # Make it symmetric

            label_matrices[label] = correlation_matrix.copy()

            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix,
                        xticklabels=unique_stains,
                        yticklabels=unique_stains,
                        cmap='viridis',
                        annot=True,
                        fmt='.2f',
                        square=True)
            plt.title(f'Average Attention Weights - Label {label}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{fold_dir}/label_{label}_attention_heatmap_{subset_type}.png', bbox_inches='tight')
            plt.close()

        # Calculate and plot difference matrix (Label 1 - Label 0)
        if 0 in label_matrices and 1 in label_matrices:
            diff_matrix = label_matrices[1] - label_matrices[0]

            plt.figure(figsize=(12, 10))
            sns.heatmap(diff_matrix,
                        xticklabels=unique_stains,
                        yticklabels=unique_stains,
                        cmap='RdBu_r',  # Red-Blue diverging colormap
                        center=0,  # Center the colormap at 0
                        annot=True,
                        fmt='.2f',
                        square=True)

            plt.title(f'Difference in Mean Attention Weights (Label 1 - Label 0)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{fold_dir}/attention_difference_heatmap_{subset_type}.png', bbox_inches='tight')
            plt.close()

