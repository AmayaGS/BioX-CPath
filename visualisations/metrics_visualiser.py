import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class MetricsVisualiser:
    def __init__(self, logger):
        self.setup_plot_style()
        self.logger = logger

    def setup_plot_style(self):
        """Set up consistent plotting style across all visualizations."""
        plt.style.use('default')
        # Set seaborn style base
        sns.set_style("whitegrid")

        # Custom parameters
        plt.rcParams.update({
            # Figure and axes backgrounds
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            # Grid styling
            'grid.color': '#E5E5E5',
            'grid.linestyle': '-',
            'grid.alpha': 0.5,
            'axes.grid': True,
            'axes.grid.which': 'major',
            'axes.grid.axis': 'y',  # Only horizontal gridlines
            # Spine styling
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.edgecolor': '#666666',
            # Tick styling
            'xtick.color': '#666666',
            'ytick.color': '#666666',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
        })

    def plot_metrics(self, args, all_metrics, fold_dir):
        """Main function to generate all visualizations."""

        self.logger.info(f"Generating stain importance visualizations...")

        # Patient-specific visualizations
        self.generate_patient_reports(args, all_metrics, fold_dir)

        self.plot_patient_specific_metrics(args, all_metrics, fold_dir)

        # Aggregate visualizations
        self.plot_aggregate_metrics(args, all_metrics, fold_dir)

    #  Patient-specific visualizations
    ## -----------------------------------------------------------------------------------------------------------------

    def generate_patient_reports(self, args, all_metrics, fold_dir):
        """Generate detailed text reports for each patient."""
        stain_names = {v: k for k, v in args.stain_types.items()}
        label_names = args.label_dict

        ground_truth_labels = []
        predicted_labels = []

        for patient_id, patient_data in all_metrics.items():
            label = patient_data[0]
            predicted = patient_data[1]
            stain_attentions = patient_data[2]
            layer_attentions = patient_data[3]
            ground_truth_labels.append(label)
            predicted_labels.append(predicted)

            report = []
            report.append(f"Patient Report: {patient_id}")
            report.append("=" * 50)
            report.append(f"\nGround Truth: {label_names[str(label)]}")
            report.append(f"Predicted Label: {label_names[str(predicted)]}")
            report.append("-" * 50)

            # Add stain importance per layer
            report.append("\nStain Attention by Layer:")
            report.append("-" * 25)

            # Create header
            header = ["Layer"] + [stain_names[s] for s in sorted(stain_names.keys())]
            report.append("\t".join(header))

            # Add values
            for layer_idx, stain_dict in enumerate(stain_attentions):
                values = [f"Layer {layer_idx + 1}"]
                for stain_num in sorted(stain_names.keys()):
                    value = stain_dict.get(stain_num, 0.0)
                    values.append(f"{float(value):.3f}")
                report.append("\t".join(values))

            # Add layer attention values
            report.append("\nLayer Attention Values:")
            report.append("-" * 25)
            for layer_idx, attention in enumerate(layer_attentions):
                report.append(f"Layer {layer_idx + 1}: {float(attention):.3f}")

            # Save report
            report_path = os.path.join(fold_dir, str(patient_id))
            os.makedirs(report_path, exist_ok=True)
            with open(os.path.join(report_path, f"{patient_id}_report.txt"), 'w') as f:
                f.write("\n".join(report))

        conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
        class_report = classification_report(ground_truth_labels, predicted_labels, zero_division=0)
        accuracy = accuracy_score(ground_truth_labels, predicted_labels)

        with open(os.path.join(fold_dir, f"accuracy_report.txt"), 'w') as f:
            f.write(f"Confusion Matrix: \n{conf_matrix}\n")
            f.write(f"Classification Report: \n{class_report}\n")
            f.write(f"Accuracy: \n{accuracy}\n")


    def plot_patient_layer_attention(self, args, patient_id, patient_data, output_dir):
        """Generate layer attention plot for a single patient."""
        label = patient_data[0]
        layer_attentions = patient_data[3]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot setup with consistent color scheme
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(layer_attentions)))
        layers = [f'Layer {i + 1}' for i in range(len(layer_attentions))]
        values = [float(x) for x in layer_attentions]

        # Create bars
        bars = ax.bar(range(len(values)), values,
                      width=0.3, color=colors,
                      edgecolor='black', linewidth=1)

        # Customize plot
        ax.set_title(f'Patient {patient_id} - {args.label_dict[str(label)]}',
                     fontsize=11, fontweight='bold', y=0.95)
        ax.set_ylabel('Layer Attention Score', fontsize=10)
        ax.set_ylim(0, 1.15)  # Dynamic y-limit

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.3f}', ha='center', va='bottom')

        # Finalize and save
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "layer_attention.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_patient_stain_attention(self, args, patient_id, patient_data, output_dir):
        """Generate stain attention visualization for each layer of a patient."""
        label = patient_data[0]
        stain_attentions = patient_data[2]

        # Create figure with shared y-axis label
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(len(stain_attentions), 1, hspace=0.2)
        axs = [fig.add_subplot(gs[i]) for i in range(len(stain_attentions))]

        # Add common y-axis label
        fig.text(-0.02, 0.5, 'Stain Attention Score', va='center', rotation='vertical', fontsize=12)

        # Title
        fig.suptitle(f'Patient {patient_id} - {args.label_dict[str(label)]}',
                     fontsize=14, fontweight='bold', y=0.95)

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}
        stain_order = sorted([k for k, v in stain_names.items() if v != 'NA'])

        # Plot each layer
        for layer_idx, stain_dict in enumerate(stain_attentions):
            ax = axs[layer_idx]

            # Prepare data
            values = [0.0] * len(stain_order)
            colors = [args.stain_colors[stain_names[s]] for s in stain_order]

            # Fill in actual values
            for stain_num, value in stain_dict.items():
                if stain_names[stain_num] != 'NA':
                    idx = stain_order.index(stain_num)
                    values[idx] = float(value)

            # Create bars
            x_positions = range(len(stain_order))
            bars = ax.bar(x_positions, values, width=0.3,
                          color=colors, alpha=0.7,
                          edgecolor='black', linewidth=1)

            # Customize plot
            ax.text(-0.1, 0.5, f'Layer {layer_idx + 1}',
                    transform=ax.transAxes, fontsize=11,
                    rotation='vertical', verticalalignment='center')
            ax.set_ylim(0, 1.15)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=11)

            # Configure axes
            ax.set_xticks(x_positions)
            ax.set_xticklabels([stain_names[s] for s in stain_order])
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.savefig(os.path.join(output_dir, "stain_attention.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_patient_stain_z_score(self, args, patient_id, patient_data, output_dir):
        """Generate stain attention visualization for each layer of a patient."""
        label = patient_data[0]
        stain_attentions = patient_data[5]

        # Create figure with shared y-axis label
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(len(stain_attentions), 1, hspace=0.2)
        axs = [fig.add_subplot(gs[i]) for i in range(len(stain_attentions))]

        # Add common y-axis label
        fig.text(-0.02, 0.5, 'Stain Attention Z-Score', va='center', rotation='vertical', fontsize=12)

        # Title
        fig.suptitle(f'Patient {patient_id} - {args.label_dict[str(label)]}',
                     fontsize=14, fontweight='bold', y=0.95)

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}
        stain_order = sorted([k for k, v in stain_names.items() if v != 'NA'])

        # Plot each layer
        for layer_idx, stain_dict in enumerate(stain_attentions):
            ax = axs[layer_idx]

            # Prepare data
            values = [0.0] * len(stain_order)
            colors = [args.stain_colors[stain_names[s]] for s in stain_order]

            # Fill in actual values
            for stain_num, value in stain_dict.items():
                if stain_names[stain_num] != 'NA':
                    idx = stain_order.index(stain_num)
                    values[idx] = float(value)

            # Create bars
            x_positions = range(len(stain_order))
            bars = ax.bar(x_positions, values, width=0.3,
                          color=colors, alpha=0.7,
                          edgecolor='black', linewidth=1)

            # Customize plot
            ax.text(-0.1, 0.5, f'Layer {layer_idx + 1}',
                    transform=ax.transAxes, fontsize=11,
                    rotation='vertical', verticalalignment='center')

            min_value = min(values)
            if min_value < 0:
                # Add extra space below the x-axis (20% of the range or at least 0.2 units)
                y_min = min_value * 1.5 if min_value < -1 else min_value - 0.5
                y_max = max(values) * 1.5
            ax.set_ylim(y_min, y_max)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=11)
                if height < 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height - 0.1,
                            f'{height:.3f}',
                            ha='center', va='top', fontsize=11)

            # Add horizontal line at y=0
            ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.8, alpha=0.5)

            # Configure axes
            ax.set_xticks(x_positions)
            ax.set_xticklabels([stain_names[s] for s in stain_order])
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.savefig(os.path.join(output_dir, "stain_attention_z_score.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # def plot_patient_entropy_scores(self, args, patient_id, patient_data, output_dir):
    #     """Generate entropy score visualization for a patient."""
    #     label = patient_data[0]
    #     entropy_scores = patient_data[4]
    #
    #     # Create figure
    #     fig = plt.figure(figsize=(15, 5 * args.num_layers))
    #     gs = fig.add_gridspec(len(entropy_scores), 1, hspace=0.4)
    #     axs = [fig.add_subplot(gs[i]) for i in range(len(entropy_scores))]
    #
    #     # Add common y-axis label
    #     fig.text(0.04, 0.5, 'Entropy Score', va='center', rotation='vertical', fontsize=12)
    #
    #     # Title
    #     fig.suptitle(f'Entropy Scores\nPatient {patient_id} - {args.label_dict[str(label)]}',
    #                  fontsize=14, fontweight='bold', y=0.95)
    #
    #     # Get stain information
    #     stain_names = {v: k for k, v in args.stain_types.items()}
    #     stain_order = ['global'] + sorted([k for k, v in stain_names.items() if v != 'NA'])
    #     colors = ['#888888'] + [args.stain_colors[stain_names[s]] for s in stain_order[1:]]
    #
    #     # Find global max for consistent scaling
    #     max_value = max(max(score.get('global', [0])[0] if isinstance(score.get('global', [0]), list)
    #                         else score.get('global', [0]) for score in entropy_scores),
    #                     max(max(score.get(stain, [0])[0] if isinstance(score.get(stain, [0]), list)
    #                             else score.get(stain, [0]) for stain in stain_order[1:])
    #                         for score in entropy_scores if any(s in score for s in stain_order[1:])))
    #
    #     # Plot each layer
    #     for layer_idx, layer_dict in enumerate(entropy_scores):
    #         ax = axs[layer_idx]
    #
    #         # Prepare data
    #         values = []
    #         # Get global score
    #         global_scores = layer_dict.get('global', [0.0])
    #         values.append(np.mean(global_scores) if isinstance(global_scores, list) else global_scores)
    #
    #         # Get stain scores
    #         for stain in stain_order[1:]:
    #             stain_scores = layer_dict.get(stain, [0.0])
    #             values.append(np.mean(stain_scores) if isinstance(stain_scores, list) else stain_scores)
    #
    #         # Create bars
    #         bars = ax.bar(range(len(values)), values, width=0.6,
    #                       color=colors, alpha=0.7,
    #                       edgecolor='black', linewidth=1)
    #
    #         # Customize plot
    #         ax.text(-0.2, 0.5, f'Layer {layer_idx + 1}',
    #                 transform=ax.transAxes, fontsize=11,
    #                 verticalalignment='center')
    #         ax.set_ylim(0, max_value * 1.15)
    #
    #         # Add value labels
    #         for bar in bars:
    #             height = bar.get_height()
    #             if height > 0:
    #                 ax.text(bar.get_x() + bar.get_width() / 2., height,
    #                         f'{height:.3f}',
    #                         ha='center', va='bottom', fontsize=10)
    #
    #         # Configure axes
    #         labels = ['Global'] + [stain_names[s] for s in stain_order[1:]]
    #         ax.set_xticks(range(len(labels)))
    #         ax.set_xticklabels(labels)
    #         ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    #
    #     plt.savefig(os.path.join(output_dir, "entropy_scores.png"),
    #                 dpi=300, bbox_inches='tight')
    #     plt.close()

    def plot_patient_edge_importance(self, args, patient_id, patient_data, output_dir):
        """Generate edge importance visualization for a patient."""
        label = patient_data[0]
        graph_metrics = patient_data[5][patient_id][1]

        # Create figure
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)

        # Get edge types and prepare data
        edge_types = list(args.edge_types.keys())
        layer_nums = range(1, args.num_layers + 1)
        data = {edge_type: [] for edge_type in edge_types}

        # Collect data for each layer
        for layer_idx in range(1, args.num_layers + 1):
            layer_key = f'Layer_{layer_idx}'
            layer_data = graph_metrics[layer_key]['edge_type_importance']
            for edge_type in edge_types:
                data[edge_type].append(layer_data[edge_type])

        # Create stacked bar plot
        bottom = np.zeros(args.num_layers)

        for edge_type in edge_types:
            plt.bar(layer_nums, data[edge_type], bottom=bottom, width=0.3,
                    label=edge_type, color=args.edge_colors[edge_type],
                    alpha=0.7, edgecolor='black', linewidth=1)

            # Add percentage labels
            for idx, value in enumerate(data[edge_type]):
                height = value / 2 + bottom[idx]
                plt.text(layer_nums[idx], height, f'{value:.1%}',
                         ha='center', va='center')

            bottom += data[edge_type]

        # Customize plot
        plt.title(f'Edge Type Importance\nPatient {patient_id} - {args.label_dict[str(label)]}',
                  pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Layer', fontsize=10)
        plt.ylabel('Edge Type Importance', fontsize=10)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add grid and legend
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Edge Types', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set x-ticks
        plt.xticks(layer_nums)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'edge_type_importance.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_patient_specific_metrics(self, args, all_metrics, fold_dir):
        """
        Generate all patient-specific visualizations.

        Args:
            args: Configuration arguments
            all_metrics: Dictionary of all patient metrics
            fold_dir: Base directory for saving visualizations
        """
        # Track successful and failed visualizations for reporting
        visualization_stats = {
            'total': len(all_metrics),
            'successful': 0,
            'failed': [],
            'warnings': []
        }

        for patient_id, patient_data in all_metrics.items():
            try:
                # Create patient-specific directory
                patient_dir = os.path.join(fold_dir, str(patient_id))
                os.makedirs(patient_dir, exist_ok=True)

                # Validate patient data structure
                if not self._validate_patient_data(patient_data):
                    visualization_stats['failed'].append({
                        'patient_id': patient_id,
                        'reason': 'Invalid data structure'
                    })
                    continue

                # Generate all visualizations for this patient
                visualization_functions = [
                    (self.plot_patient_layer_attention, "layer attention"),
                    (self.plot_patient_layer_entropy, "layer entropy"),
                    (self.plot_patient_stain_attention, "stain attention"),
                    (self.plot_patient_stain_z_score, "stain z-score"),
                    (self.plot_patient_stain_entropy, "stain entropy"),
                    (self.plot_patient_edge_importance, "edge importance")
                ]

                for viz_func, viz_name in visualization_functions:
                    try:
                        viz_func(args, patient_id, patient_data, patient_dir)
                    except Exception as e:
                        visualization_stats['warnings'].append({
                            'patient_id': patient_id,
                            'visualization': viz_name,
                            'error': str(e)
                        })

                # Generate summary plots if patient has multiple stains
                if self._has_multiple_stains(patient_data):
                    self._generate_multi_stain_summaries(
                        args, patient_id, patient_data, patient_dir)

                visualization_stats['successful'] += 1

            except Exception as e:
                visualization_stats['failed'].append({
                    'patient_id': patient_id,
                    'reason': str(e)
                })

        # Generate visualization report
        self._save_visualization_report(visualization_stats, fold_dir)

    def _validate_patient_data(self, patient_data):
        """
        Validate the structure of patient data.

        Expected structure:
        - patient_data[0]: label
        - patient_data[1]: predicted label
        - patient_data[2]: stain attentions
        - patient_data[3]: layer attentions
        - patient_data[4]: entropy scores
        - patient_data[5]: stain_z_scores
        - patient_data[6]: graph metrics
        """
        try:
            required_lengths = {
                'basic': 7,  # Total number of main elements
                'stain_attentions': len(patient_data[3]),  # Should match number of layers
                'layer_attentions': len(patient_data[3]),
                'entropy_scores': len(patient_data[3])
            }

            # Check basic structure
            if len(patient_data) < required_lengths['basic']:
                return False

            # Check consistent lengths across layer-specific data
            if not (len(patient_data[2]) == len(patient_data[3]) == len(patient_data[4])):
                return False

            # Check data types
            if not (isinstance(patient_data[0], (int, np.integer)) and
                    isinstance(patient_data[1], (int, np.integer))):
                return False

            return True

        except Exception:
            return False

    def _has_multiple_stains(self, patient_data):
        """Check if patient has multiple stain types."""
        unique_stains = set()
        for stain_dict in patient_data[2]:  # stain_attentions
            unique_stains.update(stain_dict.keys())
        return len(unique_stains) > 1

    def _generate_multi_stain_summaries(self, args, patient_id, patient_data, output_dir):
        """Generate additional visualizations for multi-stain patients."""

        # Layer-wise stain importance trends
        self._plot_stain_importance_trends(args, patient_id, patient_data, output_dir)

    def _save_visualization_report(self, stats, fold_dir):
        """Save a report of the visualization process."""
        report_path = os.path.join(fold_dir, 'visualization_report.txt')

        with open(report_path, 'w') as f:
            f.write("Visualization Report\n")
            f.write("===================\n\n")
            f.write(f"Total patients processed: {stats['total']}\n")
            f.write(f"Successfully processed: {stats['successful']}\n")
            f.write(f"Failed: {len(stats['failed'])}\n\n")

            if stats['failed']:
                f.write("Failed Patients:\n")
                for failure in stats['failed']:
                    f.write(f"- Patient {failure['patient_id']}: {failure['reason']}\n")
                f.write("\n")

            if stats['warnings']:
                f.write("Warnings:\n")
                for warning in stats['warnings']:
                    f.write(f"- Patient {warning['patient_id']}, "
                            f"{warning['visualization']}: {warning['error']}\n")

    def plot_patient_layer_entropy(self, args, patient_id, patient_data, output_dir):
        """Generate layer-wise entropy visualization for global entropy."""
        label = patient_data[0]
        entropy_scores = patient_data[4]

        # Extract global entropy scores for each layer
        global_entropy = [layer_dict['global'][0] if 'global' in layer_dict else 0
                          for layer_dict in entropy_scores]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # # Add grid first (behind the bars)
        # ax.grid(True, color='white', linewidth=1.5, zorder=0)
        # ax.set_axisbelow(True)

        # Prepare data
        layers = [f'Layer {i + 1}' for i in range(len(global_entropy))]
        x_positions = range(len(global_entropy))

        # red color gradient for bars
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(global_entropy)))

        # Create bars
        bars = ax.bar(x_positions, global_entropy,
                      width=0.6,
                      color=colors,
                      edgecolor='black',
                      linewidth=0.5,
                      zorder=2)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#222222',
                    zorder=3)

        # Customize plot
        ax.set_title(f'Patient {patient_id} - {args.label_dict[str(label)]}',
                     fontsize=11, fontweight='bold', color='#222222')
        ax.set_ylabel('Global Entropy Score', fontsize=10, color='#444444')

        # Set x-ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(layers)

        # Style ticks
        plt.setp(ax.get_xticklabels(), fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        # Set fixed x-axis limits with padding
        ax.set_xlim(-0.5, len(global_entropy) - 0.5)

        # Add some padding to y-axis for labels
        ymax = max(global_entropy)
        #ax.set_ylim(0, ymax * 1.15)
        ax.set_ylim(0, 10)

        # plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_entropy_global.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_patient_stain_entropy(self, args, patient_id, patient_data, output_dir):
        """Generate stain-wise entropy visualization for a patient."""
        label = patient_data[0]
        entropy_scores = patient_data[4]

        # Create figure with shared y-axis label
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(len(entropy_scores), 1, hspace=0.2)
        axs = [fig.add_subplot(gs[i]) for i in range(len(entropy_scores))]

        # Add common y-axis label
        fig.text(-0.02, 0.5, 'Stain Entropy Score', va='center', rotation='vertical', fontsize=12)

        # Title
        fig.suptitle(f'Patient {patient_id} - {args.label_dict[str(label)]}',
                     fontsize=14, fontweight='bold', y=0.95)

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}
        stain_order = sorted([k for k, v in stain_names.items() if v != 'NA'])

        # Plot each layer
        for layer_idx, layer_dict in enumerate(entropy_scores):
            ax = axs[layer_idx]

            # Prepare data
            values = [0.0] * len(stain_order)
            colors = [args.stain_colors[stain_names[s]] for s in stain_order]

            # Fill in actual values
            for stain in stain_order:
                if stain in layer_dict:
                    idx = stain_order.index(stain)
                    stain_scores = layer_dict[stain]
                    values[idx] = np.mean(stain_scores) if isinstance(stain_scores, list) else stain_scores

            # Create bars
            x_positions = range(len(stain_order))
            bars = ax.bar(x_positions, values,
                          width=0.3,
                          color=colors,
                          alpha=0.7,
                          edgecolor='black',
                          linewidth=1)

            # Customize plot
            ax.text(-0.1, 0.5, f'Layer {layer_idx + 1}',
                    transform=ax.transAxes, fontsize=11,
                    rotation='vertical', verticalalignment='center')
            ax.set_ylim(0, 10)  # Fixed y-limit for entropy scores

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=11)

            # Configure axes
            ax.set_xticks(x_positions)
            ax.set_xticklabels([stain_names[s] for s in stain_order])
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.savefig(os.path.join(output_dir, 'stain_entropy.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_stain_importance_trends(self, args, patient_id, patient_data, output_dir):
        """
        Plot layer-wise trends of stain importance for multi-stain patients.
        Shows how each stain's importance changes across layers.
        """
        label = patient_data[0]
        stain_attentions = patient_data[2]

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))

        # Prepare data
        layers = range(1, len(stain_attentions) + 1)

        # Track all stains present in any layer
        all_stains = set()
        for layer_data in stain_attentions:
            all_stains.update(layer_data.keys())

        # Remove NA if present
        all_stains = {s for s in all_stains if stain_names[s] != 'NA'}

        # Create trend lines for each stain
        for stain in sorted(all_stains):
            stain_values = []
            for layer_data in stain_attentions:
                value = layer_data.get(stain, 0.0)
                stain_values.append(float(value))

            # Plot trend line with markers
            ax.plot(layers, stain_values,
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=stain_names[stain],
                    color=args.stain_colors[stain_names[stain]],
                    alpha=0.7)

            # Add value labels
            for x, y in zip(layers, stain_values):
                ax.text(x, y, f'{y:.2f}',
                        ha='center', va='bottom',
                        fontsize=8)

        # Customize plot
        ax.set_title(f'Stain Importance Trends Across Layers\nPatient {patient_id} - {args.label_dict[str(label)]}',
                     pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Stain Importance Score', fontsize=10)

        # Set axis limits
        ax.set_xlim(0.5, len(layers) + 0.5)
        ax.set_ylim(0, 1.15)  # Importance scores are normalized

        # Configure grid
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Set x-ticks
        ax.set_xticks(layers)
        ax.set_xticklabels([f'Layer {i}' for i in layers])

        # Add legend
        ax.legend(title='Stain Types',
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stain_importance_trends.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


    #  Aggregated visualizations
    ## -----------------------------------------------------------------------------------------------------------------

    def plot_aggregate_metrics(self, args, all_metrics, fold_dir):
        """Generate all aggregate visualizations."""
        # Identify multi-stain patients
        multi_stain_patients = self._get_multi_stain_patients(all_metrics)

        # Generate multi-stain report
        self._generate_multi_stain_report(multi_stain_patients, fold_dir)

        # Stain-related visualizations (for all patients and multi-stain subset)
        self.plot_stain_importance_by_label(args, all_metrics, fold_dir, "all")
        self.plot_stain_z_score_by_label(args, all_metrics, fold_dir, "all")
        self.plot_stain_importance_by_layer(args, all_metrics, fold_dir, "all")
        self.plot_stain_z_score_by_layer(args, all_metrics, fold_dir, "all")
        self.plot_entropy_scores_by_label(args, all_metrics, fold_dir, 'all')
        self.plot_entropy_scores_by_layer(args, all_metrics, fold_dir, 'all')

        if multi_stain_patients:
            self.plot_stain_importance_by_label(args, multi_stain_patients, fold_dir, "multi_stain")
            self.plot_stain_z_score_by_label(args, all_metrics, fold_dir, "multi_stain")
            self.plot_stain_importance_by_layer(args, multi_stain_patients, fold_dir, "multi_stain")
            self.plot_stain_z_score_by_layer(args, multi_stain_patients, fold_dir, "multi_stain")
            self.plot_entropy_scores_by_label(args, multi_stain_patients, fold_dir, 'multi_stain')
            self.plot_entropy_scores_by_layer(args, multi_stain_patients, fold_dir, 'multi_stain')

        # Layer attention visualizations
        self.plot_layer_attention_by_label(args, all_metrics, fold_dir)

        # Edge-related visualizations
        self.plot_edge_importance_by_label(args, all_metrics, fold_dir)

    def _get_multi_stain_patients(self, all_metrics):
        """
        Identify patients with multiple stain types across layers.

        Args:
            all_metrics (dict): Dictionary of all patient metrics.

        Returns:
            dict: Dictionary containing only patients with multiple stain types.
        """
        multi_stain_patients = {}

        for patient_id, patient_data in all_metrics.items():
            # Get unique stains across all layers
            unique_stains = set()
            for stain_dict in patient_data[2]:  # stain_attentions
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

    def _generate_multi_stain_report(self, multi_stain_patients, fold_dir):
        """Generate a detailed report about multi-stain patients."""
        report_lines = [
            "Multi-stain Patient Report",
            "=======================\n",
            f"Total multi-stain patients: {len(multi_stain_patients)}",
            "\nPatient IDs and their stain combinations:"
        ]

        # Sort patients by ID for consistent reporting
        sorted_patients = sorted(multi_stain_patients.items())

        for patient_id, patient_data in sorted_patients:
            # Get unique stains across all layers
            unique_stains = set()
            for stain_dict in patient_data[2]:  # stain_attentions
                unique_stains.update(stain_dict.keys())

            stains_str = ", ".join(sorted(str(s) for s in unique_stains))
            report_lines.append(f"\nPatient {patient_id}:")
            report_lines.append(f"  Stains: {stains_str}")
            report_lines.append(f"  Label: {patient_data[0]}")

        # Save report
        report_path = os.path.join(fold_dir, 'multi_stain_report.txt')
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

    def plot_stain_importance_by_label(self, args, patient_data, fold_dir, subset_type):
        """Plot stain importance distribution by label using boxplots."""

        # Initialize data structure for each label
        label_data = {0: {}, 1: {}}

        # Collect data for each patient
        for patient_id, data in patient_data.items():
            label = data[0]
            stain_importance = data[2]
            # Process each layer's stain importance
            for layer in stain_importance:
                for stain, importance in layer.items():
                    if stain not in label_data[label]:
                        label_data[label][stain] = []
                    label_data[label][stain].append(importance.item())  # importance * layer_importance

        # Conduct significance tests between label groups for each stain
        significance_results = self._compute_group_significance(
            label_data[0], label_data[1], test_type='mann_whitney'
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Stain Attention Distribution by Label',
                     fontsize=14, fontweight='bold', y=1.05)

        # Plot for each label
        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            data = label_data[label_idx]

            # Prepare data for boxplot
            stain_order = sorted(data.keys())
            box_data = [data[stain] for stain in stain_order]
            stain_names = {v: k for k, v in args.stain_types.items()}
            labels = [stain_names[s] for s in stain_order]
            colors = [args.stain_colors[label] for label in labels]

            # Create boxplot
            bp = ax.boxplot(box_data,
                            labels=labels,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add summary statistics
            for i, (stain_idx, values) in enumerate(zip(stain_order, box_data), 1):
                mean = np.mean(values)
                std = np.std(values)
                stats_text = f'μ={mean:.2f}\nσ={std:.2f}'

                # Add significance stars if applicable and this is the second label
                if label_idx == 1 and stain_idx in significance_results:
                    stars = significance_results[stain_idx]['stars']
                    if stars:
                        stats_text = f'{stars}\n{stats_text}'

                ax.text(i, -0.2, stats_text,
                        ha='center', va='top', fontsize=10)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Stain Attention Score', fontsize=10)
            ax.set_ylim(-0.05, 1.15)

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

        # Add a legend for significance levels
        legend_text = '* p<0.05   ** p<0.01   *** p<0.001'
        fig.text(0.5, 0.01, legend_text, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'stain_importance_by_label_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')

        # Save significance test results to a CSV file
        if significance_results:
            result_file = os.path.join(fold_dir, f'stain_significance_{subset_type}.csv')
            self._save_significance_results(significance_results, stain_names, result_file)

        plt.close()

    def _compute_group_significance(self, group_a_data, group_b_data, test_type='mann_whitney'):

        import scipy.stats as stats

        significance_results = {}

        # Get all unique categories
        all_categories = sorted(set(list(group_a_data.keys()) + list(group_b_data.keys())))

        for category in all_categories:
            # Skip if the category doesn't exist in both groups
            if category not in group_a_data or category not in group_b_data:
                continue

            # Get data for both groups
            values_a = group_a_data[category]
            values_b = group_b_data[category]

            # Skip if either group has no data
            if not values_a or not values_b:
                continue

            # Perform statistical test
            if test_type == 'mann_whitney':
                # Non-parametric test (doesn't assume normal distribution)
                stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            elif test_type == 't_test':
                # Parametric test (assumes normal distribution)
                stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            # Determine significance level
            significance = False
            stars = ''

            if p_value < 0.05:
                significance = True
                if p_value < 0.001:
                    stars = '***'
                elif p_value < 0.01:
                    stars = '**'
                else:
                    stars = '*'

            # Store results
            significance_results[category] = {
                'p_value': p_value,
                'statistic': stat,
                'significant': significance,
                'stars': stars
            }

        return significance_results

    def _save_significance_results(self, significance_results, category_names, output_path):
        """Save significance test results to a CSV file."""
        with open(output_path, 'w') as f:
            f.write('Category,p-value,Significant,Significance Level\n')
            for category, result in significance_results.items():
                if category in category_names:
                    name = category_names[category]
                else:
                    name = str(category)

                stars = result['stars'] if 'stars' in result else ''
                f.write(f'{name},{result["p_value"]:.5f},{result["significant"]},{stars}\n')

    def plot_stain_z_score_by_label(self, args, patient_data, fold_dir, subset_type):
        """Plot stain importance distribution by label using boxplots."""

        # Initialize data structure for each label
        label_data = {0: {}, 1: {}}

        # Collect data for each patient
        for patient_id, data in patient_data.items():
            label = data[0]
            stain_importance = data[5]
            # Process each layer's stain importance
            for layer in stain_importance:
                for stain, importance in layer.items():
                    if stain not in label_data[label]:
                        label_data[label][stain] = []
                    label_data[label][stain].append(importance.item())  # importance * layer_importance

        # Conduct significance tests between label groups for each stain
        significance_results = self._compute_group_significance(
            label_data[0], label_data[1], test_type='mann_whitney'
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Z-score Stain Attention Distribution by Label',
                     fontsize=14, fontweight='bold', y=1.05)

        # Plot for each label
        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            data = label_data[label_idx]

            # Prepare data for boxplot
            stain_order = sorted(data.keys())
            box_data = [data[stain] for stain in stain_order]
            stain_names = {v: k for k, v in args.stain_types.items()}
            labels = [stain_names[s] for s in stain_order]
            colors = [args.stain_colors[label] for label in labels]

            # Create boxplot
            bp = ax.boxplot(box_data,
                            labels=labels,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            showfliers=False,
                            zorder=2)

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Find y-limits for this subplot based on data
            min_val = min([min(values) for values in box_data]) if box_data else 0
            max_val = max([max(values) for values in box_data]) if box_data else 1

            # Calculate y range and add padding below for statistics
            y_range = max_val - min_val
            stats_padding = y_range * 0.1  # 30% of the data range as padding

            # Set ylim with extra space at the bottom for statistics
            y_min = min_val - stats_padding
            y_max = max_val + (y_range * 0.1)  # 10% padding at the top
            #ax.set_ylim(y_min, y_max)

            # Add summary statistics
            for i, (stain_idx, values) in enumerate(zip(stain_order, box_data), 1):
                mean = np.mean(values)
                std = np.std(values)
                stats_text = f'μ={mean:.2f}\nσ={std:.2f}'

                # Add significance stars if applicable and this is the second label
                if label_idx == 1 and stain_idx in significance_results:
                    stars = significance_results[stain_idx]['stars']
                    if stars:
                        stats_text = f'{stars}\n{stats_text}'

                ax.text(i, -11.5, stats_text,
                        ha='center', va='top', fontsize=10)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Stain Attention Z-score', fontsize=10)
            ax.set_ylim(-10, 8)

            ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

        # Add a legend for significance levels
        legend_text = '* p<0.05   ** p<0.01   *** p<0.001'
        fig.text(0.5, -0.01, legend_text, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'stain_z_score_by_label_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')

        # Save significance test results to a CSV file
        if significance_results:
            result_file = os.path.join(fold_dir, f'stain_z_score_{subset_type}.csv')
            self._save_significance_results(significance_results, stain_names, result_file)

        plt.close()

    def plot_stain_importance_by_layer(self, args, patient_data, fold_dir, subset_type):
        """Plot stain importance distribution by label and layer using boxplots."""
        # Initialize data structure for each label and layer

        # Initialize data structure for each label
        label_data = {0: {}, 1: {}}
        aggregated_data = {0: {}, 1: {}}
        for label in [0, 1]:
            label_data[label] = {f'Layer_{i}': {} for i in range(1, args.num_layers + 1)}

        # Collect data for each patient
        for patient_id, data in patient_data.items():
            label = data[0]
            stain_importance = data[2]

            # Process each layer's stain importance
            for i, layer in enumerate(stain_importance):
                layer_key = f'Layer_{i + 1}'
                for stain, importance in layer.items():
                    if stain not in label_data[label][layer_key]:
                        label_data[label][layer_key][stain] = []
                    label_data[label][layer_key][stain].append(importance.item())

        # Conduct significance tests between label groups for each stain

        # for i, layer in enumerate(stain_importance):
        #     layer_key = f'Layer_{i + 1}'
        #     significance_results = self._compute_group_significance(
        #     label_data[0][layer_key], label_data[1][layer_key], test_type='mann_whitney'
        # )

        # Create plot with multiple layers
        fig, axs = plt.subplots(args.num_layers, 2, figsize=(15, 5 * args.num_layers))
        fig.suptitle(f'Stain Importance Distribution by Layer ({subset_type.replace("_", " ").title()} Patients)',
                     fontsize=14, fontweight='bold', y=1.00)

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}

        # Plot for each layer and label
        for layer_idx in range(args.num_layers):
            layer_key = f'Layer_{layer_idx + 1}'

            significance_results = self._compute_group_significance(
                label_data[0][layer_key], label_data[1][layer_key], test_type='mann_whitney'
            )

            for label_idx in [0, 1]:
                ax = axs[layer_idx, label_idx]
                data = label_data[label_idx][layer_key]

                # Prepare data for boxplot
                stain_order = sorted(data.keys())
                box_data = [data[stain] for stain in stain_order]
                stain_names = {v: k for k, v in args.stain_types.items()}
                labels = [stain_names[s] for s in stain_order]
                colors = [args.stain_colors[label] for label in labels]

                # Create boxplot
                bp = ax.boxplot(box_data,
                                labels=labels,
                                patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5),
                                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Add summary statistics
                for i, (stain_idx, values) in enumerate(zip(stain_order, box_data), 1):
                    mean = np.mean(values)
                    std = np.std(values)
                    stats_text = f'μ={mean:.2f}\nσ={std:.2f}'

                    # Add significance stars if applicable and this is the second label
                    if label_idx == 1 and stain_idx in significance_results:
                        stars = significance_results[stain_idx]['stars']
                        if stars:
                            stats_text = f'{stars}\n{stats_text}'

                    ax.text(i, -0.2, stats_text,
                            ha='center', va='top', fontsize=10)

                # Customize plot
                if layer_idx == 0:
                    ax.set_title(f'{args.label_dict[str(label_idx)]}',
                                 pad=20, fontsize=12, fontweight='bold')
                if label_idx == 0:
                    ax.set_ylabel(f'Layer {layer_idx + 1}\n Stain Attention Score', fontsize=10)
                # if layer_idx == args.num_layers - 1:
                #     ax.set_xlabel('Stain Type', fontsize=10)
                ax.set_ylim(-0.05, 1.15)

                # Style improvements
                ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                ax.set_axisbelow(True)
                plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

        # Add a legend for significance levels
        legend_text = '* p<0.05   ** p<0.01   *** p<0.001'
        fig.text(0.5, -0.01, legend_text, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

                # # Customize plot
                # if layer_idx == 0:
                #     ax.set_title(f'{args.label_dict[str(label_idx)]}',
                #                  pad=20, fontsize=12, fontweight='bold')
                # if label_idx == 0:
                #     ax.set_ylabel(f'Layer {layer_idx + 1}\n Stain Importance Score', fontsize=10)
                # if layer_idx == args.num_layers - 1:
                #     ax.set_xlabel('Stain Type', fontsize=10)
                # ax.set_ylim(-0.05, 1.15)
                #
                # # Style improvements
                # ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                # ax.set_axisbelow(True)
                # plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'stain_importance_by_layer_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_stain_z_score_by_layer(self, args, patient_data, fold_dir, subset_type):
        """Plot stain z-score distribution by label and layer using boxplots."""
        # Initialize data structure for each label and layer
        label_data = {0: {}, 1: {}}
        for label in [0, 1]:
            label_data[label] = {f'Layer_{i}': {} for i in range(1, args.num_layers + 1)}

        # Collect data for each patient
        for patient_id, data in patient_data.items():
            label = data[0]
            stain_z_scores = data[5]  # Using z-scores instead of raw importance

            # Process each layer's stain z-scores
            for i, layer in enumerate(stain_z_scores):
                layer_key = f'Layer_{i + 1}'
                for stain, z_score in layer.items():
                    if stain not in label_data[label][layer_key]:
                        label_data[label][layer_key][stain] = []
                    label_data[label][layer_key][stain].append(z_score.item())

        # Create plot with multiple layers
        fig, axs = plt.subplots(args.num_layers, 2, figsize=(15, 5 * args.num_layers))

        fig.suptitle(
            f'Z-score Stain Attention Distribution by Layer ({subset_type.replace("_", " ").title()} Patients)',
            fontsize=14, fontweight='bold', y=1.00)

        # Get stain information
        stain_names = {v: k for k, v in args.stain_types.items()}

        # Plot for each layer and label
        for layer_idx in range(args.num_layers):
            layer_key = f'Layer_{layer_idx + 1}'

            # Compute significance tests between label groups for this layer
            significance_results = self._compute_group_significance(
                label_data[0][layer_key], label_data[1][layer_key], test_type='mann_whitney'
            )

            for label_idx in [0, 1]:
                ax = axs[layer_idx, label_idx]
                data = label_data[label_idx][layer_key]

                # Prepare data for boxplot
                stain_order = sorted(data.keys())
                box_data = [data[stain] for stain in stain_order]
                labels = [stain_names[s] for s in stain_order]
                colors = [args.stain_colors[label] for label in labels]

                # Create boxplot
                bp = ax.boxplot(box_data,
                                labels=labels,
                                patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5),
                                showfliers=False,
                                zorder=2)

                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Find y-limits for this subplot based on data
                min_val = min([min(values) for values in box_data]) if box_data else 0
                max_val = max([max(values) for values in box_data]) if box_data else 1

                # Calculate y range and add padding below for statistics
                y_range = max_val - min_val
                stats_padding = y_range * 0.3  # 30% of the data range as padding

                # Set ylim with extra space at the bottom for statistics
                y_min = min_val - stats_padding
                y_max = max_val + (y_range * 0.1)  # 10% padding at the top
                ax.set_ylim(-12, 10)

                # Add summary statistics
                for i, (stain_idx, values) in enumerate(zip(stain_order, box_data), 1):
                    mean = np.mean(values)
                    std = np.std(values)
                    stats_text = f'μ={mean:.2f}\nσ={std:.2f}'

                    # Add significance stars if applicable and this is the second label
                    if label_idx == 1 and stain_idx in significance_results:
                        stars = significance_results[stain_idx]['stars']
                        if stars:
                            stats_text = f'{stars}\n{stats_text}'

                    # Position text at a fixed distance below the plot area
                    text_y_pos = min_val - (stats_padding * 0.5)  # Position halfway into the padding area
                    ax.text(i, -14.5, stats_text,
                            ha='center', va='center', fontsize=10)

                # Customize plot
                if layer_idx == 0:
                    ax.set_title(f'{args.label_dict[str(label_idx)]}',
                                 pad=20, fontsize=12, fontweight='bold')
                if label_idx == 0:
                    ax.set_ylabel(f'Layer {layer_idx + 1}\nStain Attention Z-score', fontsize=10)

                # Add horizontal line at y=0 to distinguish positive/negative values
                ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

                # Style improvements
                ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                ax.set_axisbelow(True)
                plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

        # Add a legend for significance levels
        legend_text = '* p<0.05   ** p<0.01   *** p<0.001'
        fig.text(0.5, 0.01, legend_text, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend
        plt.savefig(os.path.join(fold_dir, f'stain_z_score_by_layer_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')

        # # Save significance test results to a CSV file for the last layer
        # layer_key = f'Layer_{args.num_layers}'
        # significance_results = self._compute_group_significance(
        #     label_data[0][layer_key], label_data[1][layer_key], test_type='mann_whitney'
        # )
        # if significance_results:
        #     result_file = os.path.join(fold_dir, f'stain_z_score_by_layer_{subset_type}.csv')
        #     self._save_significance_results(significance_results, stain_names, result_file)

        plt.close()

    def plot_entropy_scores_by_label(self, args, all_metrics, fold_dir, subset_type):
        """Plot entropy scores distribution by label using boxplots."""
        # Initialize data structures for global and stain-specific entropy
        label_data = {
            0: {'global': [], 'stains': {}},
            1: {'global': [], 'stains': {}}
        }

        # Collect data
        for patient_id, patient_data in all_metrics.items():
            label = patient_data[0]
            entropy_scores = patient_data[4]

            for layer_dict in entropy_scores:
                # Handle global entropy
                if 'global' in layer_dict:
                    label_data[label]['global'].extend(
                        layer_dict['global'] if isinstance(layer_dict['global'], list)
                        else [layer_dict['global']]
                    )

                # Handle stain-specific entropy
                for stain_type, scores in layer_dict.items():
                    if stain_type != 'global':
                        if stain_type not in label_data[label]['stains']:
                            label_data[label]['stains'][stain_type] = []
                        label_data[label]['stains'][stain_type].extend(
                            scores if isinstance(scores, list) else [scores]
                        )

        self._plot_stain_entropy_by_label(args, label_data, fold_dir, subset_type)

    def _plot_stain_entropy_by_label(self, args, label_data, fold_dir, subset_type):
        """Plot stain-specific entropy scores by label."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        stain_names = {v: k for k, v in args.stain_types.items()}

        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            stain_entropy = label_data[label_idx]['stains']

            # Prepare data for boxplot
            stain_order = sorted(stain_entropy.keys())
            box_data = [stain_entropy[stain] for stain in stain_order]
            labels = [stain_names[s] for s in stain_order]
            colors = [args.stain_colors[label] for label in labels]

            # Create boxplot
            bp = ax.boxplot(box_data,
                            labels=labels,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add summary statistics
            for i, values in enumerate(box_data, 1):
                mean = np.mean(values)
                std = np.std(values)
                stats_text = f'μ={mean:.2f}\nσ={std:.2f}'
                ax.text(i, -1.0, stats_text,
                        ha='center', va='top', fontsize=10)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Stain Entropy Score', fontsize=10)

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
            ax.set_ylim(-0.1, 10)

        plt.suptitle(f'Stain Entropy Distribution by Label ({subset_type.replace("_", " ").title()} Patients)',
                     fontsize=14, fontweight='bold', y=1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'stain_entropy_by_label_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_entropy_scores_by_layer(self, args, all_metrics, fold_dir, subset_type):
        """Plot entropy scores distribution by layer and label."""
        # Initialize data structure for each label and layer
        label_data = {0: {}, 1: {}}
        for label in [0, 1]:
            label_data[label] = {
                'global': [[] for _ in range(args.num_layers)],
                'stains': {}
            }

        # Collect data
        for patient_id, patient_data in all_metrics.items():
            label = patient_data[0]
            entropy_scores = patient_data[4]

            # Process each layer
            for layer_idx, layer_dict in enumerate(entropy_scores):
                # Handle global entropy
                if 'global' in layer_dict:
                    scores = layer_dict['global']
                    scores = scores if isinstance(scores, list) else [scores]
                    label_data[label]['global'][layer_idx].extend(scores)

                # Handle stain-specific entropy
                for stain_type, scores in layer_dict.items():
                    if stain_type != 'global':
                        if stain_type not in label_data[label]['stains']:
                            label_data[label]['stains'][stain_type] = [[] for _ in range(args.num_layers)]
                        scores = scores if isinstance(scores, list) else [scores]
                        label_data[label]['stains'][stain_type][layer_idx].extend(scores)

        # Create separate plots for global and stain entropy
        self._plot_layer_wise_global_entropy(args, label_data, fold_dir, subset_type)
        self._plot_layer_wise_stain_entropy(args, label_data, fold_dir, subset_type)

    def _plot_layer_wise_global_entropy(self, args, label_data, fold_dir, subset_type):
        """Plot layer-wise global entropy distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Define color gradient for layers
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, args.num_layers))

        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            global_entropy = label_data[label_idx]['global']

            # Create boxplot
            bp = ax.boxplot(global_entropy,
                            labels=[f'Layer {i + 1}' for i in range(args.num_layers)],
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add summary statistics
            for i, layer_data in enumerate(global_entropy, 1):
                if layer_data:  # Only add stats if we have data
                    mean = np.mean(layer_data)
                    std = np.std(layer_data)
                    counts = len(layer_data)
                    stats_text = f'μ={mean:.2f}\nσ={std:.2f}'
                    ax.text(i, -1.0, stats_text,
                            ha='center', va='top', fontsize=8)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Global Entropy Score', fontsize=10)

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

            ax.set_ylim(-0.1, 10)

        plt.suptitle(f'Layer-wise Global Entropy Distribution by Label ({subset_type.replace("_", " ").title()} Patients)',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'global_entropy_by_layer_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_layer_wise_stain_entropy(self, args, label_data, fold_dir, subset_type):
        """Plot layer-wise stain entropy distribution."""
        stain_names = {v: k for k, v in args.stain_types.items()}

        # Create a single figure with subplots arranged in a grid
        fig, axs = plt.subplots(args.num_layers, 2,
                                figsize=(15, 5 * args.num_layers))
        if args.num_layers == 1:
            axs = axs.reshape(1, 2)

        # Plot each layer
        for layer_idx in range(args.num_layers):
            # Plot each label
            for label_idx in [0, 1]:
                ax = axs[layer_idx, label_idx]

                # Skip if no stain data for this label
                if not label_data[label_idx]['stains']:
                    ax.text(0.5, 0.5, 'No data available',
                            ha='center', va='center')
                    continue

                # Prepare data for boxplot
                stain_order = sorted(label_data[label_idx]['stains'].keys())
                box_data = [label_data[label_idx]['stains'][stain][layer_idx]
                            for stain in stain_order]
                labels = [stain_names[s] for s in stain_order]
                colors = [args.stain_colors[label] for label in labels]

                # Create boxplot
                bp = ax.boxplot(box_data,
                                labels=labels,
                                patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5),
                                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Customize plot
                if layer_idx == 0:
                    ax.set_title(f'{args.label_dict[str(label_idx)]}',
                                 pad=20, fontsize=12, fontweight='bold')
                if label_idx == 0:
                    ax.set_ylabel(f'Layer {layer_idx + 1}\n Stain Entropy Score', fontsize=10)
                if layer_idx == args.num_layers - 1:
                    ax.set_xlabel('Stain Type', fontsize=10)

                ax.set_ylim(-0.1, 10)

                # Style improvements
                ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                ax.set_axisbelow(True)
                plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        plt.suptitle(f'Layer-wise Stain Entropy Distribution by Label ({subset_type.replace("_", " ").title()} Patients)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'stain_entropy_by_layer_{subset_type}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_layer_attention_by_label(self, args, all_metrics, fold_dir):
        """Plot layer attention scores distribution by label."""
        # Initialize data structures
        label_data = {0: [], 1: []}

        # Collect layer attention scores
        for patient_id, patient_data in all_metrics.items():
            label = patient_data[0]
            layer_attention = patient_data[3]
            label_data[label].append(layer_attention)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Define color gradient for layers
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, args.num_layers))

        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            scores = label_data[label_idx]

            # Transpose to get layer-wise grouping
            box_data = list(map(list, zip(*scores)))

            # Create boxplot
            bp = ax.boxplot(box_data,
                            labels=[f'Layer {i + 1}' for i in range(args.num_layers)],
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add summary statistics
            for i, layer_data in enumerate(box_data, 1):
                mean = np.mean(layer_data)
                std = np.std(layer_data)
                stats_text = f'μ={mean:.2f}\nσ={std:.2f}'
                ax.text(i, -0.2, stats_text,
                        ha='center', va='top', fontsize=8)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Layer Attention Score', fontsize=10)

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

            # Set y-axis limits
            ax.set_ylim(-0.05, 1.15)  # Allow space for statistics below

        plt.suptitle('Layer Attention Distribution by Label',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'layer_attention_by_label.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_edge_importance_by_label(self, args, all_metrics, fold_dir):
        """Plot edge importance distribution by label."""
        # Initialize data structure
        label_data = {0: {}, 1: {}}

        # Collect data
        for patient_id, patient_data in all_metrics.items():
            label = patient_data[0]
            graph_metrics = patient_data[6][patient_id][1]

            for layer_key in [f'Layer_{i}' for i in range(1, args.num_layers + 1)]:
                edge_importance = graph_metrics[layer_key]['edge_type_importance']
                for edge_type, importance in edge_importance.items():
                    if edge_type not in label_data[label]:
                        label_data[label][edge_type] = []
                    label_data[label][edge_type].append(importance)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for label_idx in [0, 1]:
            ax = ax1 if label_idx == 0 else ax2
            data = label_data[label_idx]

            # Prepare data for boxplot
            edge_types = list(data.keys())
            box_data = [data[edge_type] for edge_type in edge_types]
            colors = [args.edge_colors[edge_type] for edge_type in edge_types]

            # Create boxplot
            bp = ax.boxplot(box_data,
                            labels=edge_types,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add summary statistics
            for i, values in enumerate(box_data, 1):
                mean = np.mean(values)
                std = np.std(values)
                counts = len(values)
                stats_text = f'μ={mean:.2f}\nσ={std:.2f}'
                ax.text(i, -0.05, stats_text,
                        ha='center', va='top', fontsize=8)

            # Customize plot
            ax.set_title(f'{args.label_dict[str(label_idx)]}',
                         pad=20, fontsize=12, fontweight='bold')
            if label_idx == 0:
                ax.set_ylabel('Edge Importance', fontsize=10)

            # Set y-axis limits
            ax.set_ylim(-0.0, 1.0)

            # Set y-axis to percentage format
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            # Style improvements
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

        plt.suptitle('Edge Type Importance Distribution by Label',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'edge_importance_by_label.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()