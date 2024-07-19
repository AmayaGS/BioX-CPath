import matplotlib.pyplot as plt
import numpy as np



def plot_training_results(results_dict, fold, save_path):

    epochs = range(1, len(results_dict['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, results_dict['train_loss'], 'r-', label='Training Loss')
    plt.plot(epochs, results_dict['val_loss'], 'b-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, results_dict['train_accuracy'], 'r-', label='Training Accuracy')
    plt.plot(epochs, results_dict['val_accuracy'], 'b-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # AUC plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, results_dict['val_auc'], 'g-', label='Validation AUC')
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

    metrics = ['loss', 'accuracy', 'auc']
    types = ['train', 'val']

    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        for t in types:
            if metric == 'auc' and t == 'train':
                continue  # Skip if we don't have training AUC
            means = np.mean([results[f'{t}_{metric}'] for results in all_results], axis=0)
            stds = np.std([results[f'{t}_{metric}'] for results in all_results], axis=0)

            plt.plot(epochs, means, '-', label=f'{t.capitalize()} {metric.capitalize()}')
            plt.fill_between(epochs, means - stds, means + stds, alpha=0.2)

        plt.title(f'Average {metric.capitalize()} Across Folds')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/average_training_results.png')
    plt.close()