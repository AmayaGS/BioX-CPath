import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np


def plot_training_results(results_dict, fold, save_path):

    epochs = range(1, len(results_dict['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, results_dict['train_loss'], 'r-', label='Training Loss')
    plt.plot(epochs, results_dict['val_loss'], 'b-.', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, results_dict['train_accuracy'], 'r-', label='Training Accuracy')
    plt.plot(epochs, results_dict['val_accuracy'], 'b-.', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # AUC plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, results_dict['val_auc'], 'b-.', label='Validation AUC')
    plt.title(f'Validation AUC - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/training_results_fold_{fold}.png')
    plt.close()


def plot_averaged_results(all_results, save_path):
    num_epochs = len(all_results[0]['train_loss'])
    epochs = range(1, num_epochs + 1)

    metrics = ['loss', 'accuracy', 'AUC']
    types = ['train', 'val']

    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        for t in types:
            if metric == 'AUC' and t == 'train':
                continue  # Skip if we don't have training AUC
            means = np.mean([results[f'{t}_{metric.lower()}'] for results in all_results], axis=0)
            stds = np.std([results[f'{t}_{metric.lower()}'] for results in all_results], axis=0)

            if t == 'train':
                line_style = '-'
                color = 'red'
            else:  # val
                line_style = '-.'
                color = 'blue'

            label = f'{t.capitalize()} {metric if metric == "AUC" else metric.capitalize()}'
            plt.plot(epochs, means, line_style, color=color, label=label)
            plt.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2)

        plt.title(f'Average {metric if metric == "AUC" else metric.capitalize()} Across Folds')
        plt.ylabel(metric if metric == "AUC" else metric.capitalize())
        plt.xlabel('Epochs')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'{save_path}/average_training_results.png')
    plt.close()



def plot_roc_curve(results_dict, n_classes, fold, save_path):
    plt.figure(figsize=(10, 8))

    all_labels = results_dict['all_labels']
    all_probs = results_dict['all_probs']

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='b', lw=2, linestyle='-.',
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        binary_labels = label_binarize(all_labels, classes=range(n_classes))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Area under the ROC Curve - Fold {fold}')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f'{save_path}/AUROC_{fold}.png')
    plt.close()


def plot_average_roc_curve(all_results, n_classes, save_path):
    plt.figure(figsize=(10, 8))

    if not all_results:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    else:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        for result in all_results:
            all_labels = result['all_labels']
            all_probs = result['all_probs']

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            else:
                fpr, tpr, _ = roc_curve(all_labels, all_probs.ravel())

            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2,
                 alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='b', alpha=.2, label=r'± 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig(f"{save_path}/average_roc_curve.png")
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve


def plot_average_pr_curve(all_results, n_classes, save_path):
    plt.figure(figsize=(10, 8))

    if not all_results:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    else:
        precisions = []
        recalls = []
        aucs = []
        total_positives = 0
        total_samples = 0

        for result in all_results:
            all_labels = result['all_labels']
            all_probs = result['all_probs']

            if n_classes == 2:
                precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
            else:
                precision, recall, _ = precision_recall_curve(all_labels, all_probs.ravel())

            precisions.append(precision)
            recalls.append(recall)
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

            # Count positives for chance line
            total_positives += np.sum(all_labels)
            total_samples += len(all_labels)

        # Interpolate all precision curves to a common set of recall points
        mean_recall = np.linspace(0, 1, 100)
        interp_precisions = []
        for precision, recall in zip(precisions, recalls):
            interp_precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])

        mean_precision = np.mean(interp_precisions, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_recall, mean_precision, color='b', label=f'Mean PR curve (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                 lw=2, alpha=.8)

        std_precision = np.std(interp_precisions, axis=0)
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
        plt.fill_between(mean_recall, precision_lower, precision_upper, color='b', alpha=.2, label=r'± 1 std. dev.')

        # Add chance line
        chance_level = total_positives / total_samples
        plt.plot([0, 1], [chance_level, chance_level], linestyle='--', color='black', label=f'Chance ({chance_level:.2f})',
                 alpha=0.8)

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.savefig(f"{save_path}/average_pr_curve.png")
    plt.close()