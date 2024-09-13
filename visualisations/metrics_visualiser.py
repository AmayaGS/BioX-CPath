import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MetricsVisualiser:
    def __init(self, args):
        self.args = args

    def plot_metrics(self, all_metrics, fold_dir):
        self.plot_stain_importance(all_metrics, fold_dir)
        self.plot_edge_type_importance(all_metrics, fold_dir)
        self.plot_overall_stain_importance(all_metrics, fold_dir)
        self.plot_stain_importance_by_label(all_metrics, fold_dir)
        self.plot_edge_importance(all_metrics, fold_dir)
        self.plot_stain_edge_correlation(all_metrics, fold_dir)
        self.plot_importance_trends(all_metrics, fold_dir)


    def plot_stain_importance(self, all_metrics, fold_dir):
        fig, axs = plt.subplots(len(all_metrics), 1, figsize=(10, 5 * len(all_metrics)))
        fig.suptitle('Stain Importance Across Layers')

        for i, (patient_id, patient_data) in enumerate(all_metrics.items()):
            layers = patient_data[patient_id][1]
            stain_data = {stain: [] for stain in layers['Layer_1']['stain_importance'].keys()}

            for layer in layers.values():
                for stain, importance in layer['stain_importance'].items():
                    stain_data[stain].append(importance)

            for stain, importances in stain_data.items():
                axs[i].plot(range(1, len(importances) + 1), importances, label=stain, marker='o')

            axs[i].set_title(f'Patient {patient_id}')
            axs[i].set_xlabel('Layer')
            axs[i].set_ylabel('Stain Importance')
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/stain_importance_across_layers_{fold_dir[-6:]}.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_edge_type_importance(self, all_metrics, fold_dir):
        fig, axs = plt.subplots(len(all_metrics), 1, figsize=(10, 5 * len(all_metrics)))
        fig.suptitle('Edge Type Importance Across Layers')

        for i, (patient_id, patient_data) in enumerate(all_metrics.items()):
            layers = patient_data[patient_id][1]
            edge_types = list(layers['Layer_1']['edge_type_importance'].keys())

            data = {edge_type: [] for edge_type in edge_types}
            for layer in layers.values():
                for edge_type, importance in layer['edge_type_importance'].items():
                    data[edge_type].append(importance)

            bottom = np.zeros(len(layers))
            for edge_type in edge_types:
                axs[i].bar(range(1, len(layers) + 1), data[edge_type], bottom=bottom, label=edge_type)
                bottom += data[edge_type]

            axs[i].set_title(f'Patient {patient_id}')
            axs[i].set_xlabel('Layer')
            axs[i].set_ylabel('Edge Type Importance')
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/edge_importance_across_layers.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_overall_stain_importance(self, all_metrics, fold_dir):
        stain_data = {}

        for patient_id, patient_data in all_metrics.items():
            layers = patient_data[patient_id][1]
            for layer in layers.values():
                for stain, importance in layer['stain_importance'].items():
                    if stain not in stain_data:
                        stain_data[stain] = []
                    stain_data[stain].append(importance)

        fig, ax = plt.subplots(figsize=(10, 6))

        box_data = [stain_data[stain] for stain in stain_data]
        ax.boxplot(box_data, labels=list(stain_data.keys()))

        ax.set_title('Overall Stain Importance Across All Patients and Layers')
        ax.set_ylabel('Importance')
        ax.set_xlabel('Stain Type')

        plt.savefig(f"{fold_dir}/stain_importance_overall.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_stain_importance_by_label(self, all_metrics, fold_dir):
        label_data = {0: {}, 1: {}}

        for patient_id, patient_data in all_metrics.items():
            label = patient_data[patient_id][0]['Label']
            layers = patient_data[patient_id][1]

            for layer in layers.values():
                for stain, importance in layer['stain_importance'].items():
                    if stain not in label_data[label]:
                        label_data[label][stain] = []
                    label_data[label][stain].append(importance)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for i, (label, data) in enumerate(label_data.items()):
            ax = ax1 if i == 0 else ax2
            box_data = [data[stain] for stain in data]
            ax.boxplot(box_data, labels=list(data.keys()))
            ax.set_title(f'Stain Importance for Label {label}')
            ax.set_ylabel('Importance')
            ax.set_xlabel('Stain Type')

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/stain_importance_by_label.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_edge_importance(self, all_metrics, fold_dir):
        overall_data = {}
        label_data = {0: {}, 1: {}}

        for patient_id, patient_data in all_metrics.items():
            label = patient_data[patient_id][0]['Label']
            layers = patient_data[patient_id][1]

            for layer in layers.values():
                for edge_type, importance in layer['edge_type_importance'].items():
                    if edge_type not in overall_data:
                        overall_data[edge_type] = []
                    overall_data[edge_type].append(importance)

                    if edge_type not in label_data[label]:
                        label_data[label][edge_type] = []
                    label_data[label][edge_type].append(importance)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Overall
        box_data = [overall_data[edge_type] for edge_type in overall_data]
        ax1.boxplot(box_data, labels=list(overall_data.keys()))
        ax1.set_title('Overall Edge Type Importance')
        ax1.set_ylabel('Importance')
        ax1.set_xlabel('Edge Type')

        # By label
        for i, (label, data) in enumerate(label_data.items()):
            ax = ax2 if i == 0 else ax3
            box_data = [data[edge_type] for edge_type in data]
            ax.boxplot(box_data, labels=list(data.keys()))
            ax.set_title(f'Edge Type Importance for Label {label}')
            ax.set_ylabel('Importance')
            ax.set_xlabel('Edge Type')

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/stain_importance_overall.png", dpi=300,
                    bbox_inches='tight')
        plt.close()


    def plot_stain_edge_correlation(self, all_metrics, fold_dir):
        data = []

        for patient_id, patient_data in all_metrics.items():
            label = patient_data[patient_id][0]['Label']
            layers = patient_data[patient_id][1]

            for layer_name, layer in layers.items():
                layer_data = {
                    'Patient': patient_id,
                    'Label': label,
                    'Layer': layer_name
                }
                layer_data.update(layer['stain_importance'])
                layer_data.update(layer['edge_type_importance'])
                data.append(layer_data)

        df = pd.DataFrame(data)

        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation between Stain and Edge Type Importance')
        plt.savefig(f"{fold_dir}/stain_edge_correlations.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_importance_trends(self, all_metrics, fold_dir):
        stain_data = {}
        edge_data = {}

        for patient_id, patient_data in all_metrics.items():
            layers = patient_data[patient_id][1]
            for layer_name, layer in layers.items():
                layer_num = int(layer_name.split('_')[1])

                for stain, importance in layer['stain_importance'].items():
                    if stain not in stain_data:
                        stain_data[stain] = {}
                    if layer_num not in stain_data[stain]:
                        stain_data[stain][layer_num] = []
                    stain_data[stain][layer_num].append(importance)

                for edge_type, importance in layer['edge_type_importance'].items():
                    if edge_type not in edge_data:
                        edge_data[edge_type] = {}
                    if layer_num not in edge_data[edge_type]:
                        edge_data[edge_type][layer_num] = []
                    edge_data[edge_type][layer_num].append(importance)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        for stain in stain_data:
            layers = sorted(stain_data[stain].keys())
            means = [np.mean(stain_data[stain][layer]) for layer in layers]
            ax1.plot(layers, means, marker='o', label=stain)

        ax1.set_title('Stain Importance Trends Across Layers')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Average Importance')
        ax1.legend()

        for edge_type in edge_data:
            layers = sorted(edge_data[edge_type].keys())
            means = [np.mean(edge_data[edge_type][layer]) for layer in layers]
            ax2.plot(layers, means, marker='o', label=edge_type)

        ax2.set_title('Edge Type Importance Trends Across Layers')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Average Importance')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/importance_trends.png", dpi=300,
                    bbox_inches='tight')
        plt.close()


