
import numpy as np
from collections import defaultdict
import pickle

import torch

from utils.profiling_utils import test_profiler
from utils.model_utils import process_model_output
from utils.plotting_functions_utils import *

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

use_gpu = torch.cuda.is_available()


def test_loop(args, model, test_loader, loss_fn, n_classes, logger, fold):
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_probs = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    incorrect_predictions = []

    model.eval()
    with torch.no_grad():
        for patient_ID, data_object in test_loader.dataset.items():
            test_profiler.update_peak_memory()
            data, label, _ = data_object
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            logits, Y_prob, predicted, loss = process_model_output(args, model(data, label), loss_fn)

            test_loss += loss.item()
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

            if predicted != label:
                incorrect_predictions.append((patient_ID, predicted, label))

            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            # Count correct predictions for each class
            for i in range(n_classes):
                class_correct[i] += ((predicted == label) & (label == i)).sum().item()
                class_total[i] += (label == i).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels)

    # Compute AUC
    if n_classes == 2:
        test_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=range(n_classes))
        test_auc = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')

    # Compute confusion matrix and classification report
    predicted_labels = np.argmax(all_probs, axis=1)
    conf_matrix = confusion_matrix(all_labels, predicted_labels)
    class_report = classification_report(all_labels, predicted_labels, zero_division=0)

    # Logging results
    logger.info(f"\n{'=' * 25} Split {fold} {'=' * 25}")
    logger.info("Test Results")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    for i in range(n_classes):
        logger.info(f"Class {i}: {class_correct[i]}/{class_total[i]}")
    logger.info(f"\n{conf_matrix}")
    logger.info(f"\n{class_report}")

    results_dict = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'probs': all_probs,
        'preds': predicted_labels,
        'labels': all_labels,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'incorrect_predictions': incorrect_predictions
    }

    return results_dict


def ensemble_test_results(args, results_dir, all_results, logger):
    all_probs = []
    all_preds = []
    true_labels = None

    # Load and aggregate results from each fold
    for fold in range(args.stratified_splits):
        results_path = f"{results_dir}/results_fold_{fold}.pkl"
        with open(results_path, 'rb') as f:
            fold_results = pickle.load(f)

        all_probs.append(fold_results['probs'])
        all_preds.append(fold_results['preds'])

        if true_labels is None:
            true_labels = fold_results['labels']
        else:
            assert np.array_equal(true_labels, fold_results['labels']), "Labels mismatch between folds"

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Ensemble the predictions
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, ensemble_preds)
    if args.n_classes == 2:
        auc = roc_auc_score(true_labels, ensemble_probs[:, 1])
        avg_precision = average_precision_score(true_labels, ensemble_probs[:, 1])
    else:
        auc = roc_auc_score(true_labels, ensemble_probs, multi_class='ovr', average='macro')
        avg_precision = average_precision_score(true_labels,
                                                label_binarize(ensemble_preds, classes=range(args.n_classes)),
                                                average='macro')

    conf_matrix = confusion_matrix(true_labels, ensemble_preds)
    class_report = classification_report(true_labels, ensemble_preds, zero_division=0)

    # Log results
    logger.info("\n" + "=" * 25 + " Ensemble Results " + "=" * 25)
    logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
    logger.info(f"Ensemble AUC: {auc:.4f}")
    logger.info(f"Ensemble Average Precision: {avg_precision:.4f}")
    logger.info("Ensemble Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")
    logger.info("Ensemble Classification Report:")
    logger.info(f"\n{class_report}")

    # Save ensemble results
    ensemble_results = {
        'probs': ensemble_probs,
        'preds': ensemble_preds,
        'labels': true_labels,
        'accuracy': accuracy,
        'auc': auc,
        'average_precision': avg_precision,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

    with open(f"{results_dir}/ensemble_results.pkl", 'wb') as f:
        pickle.dump(ensemble_results, f)

    logger.info(f"Ensemble results saved to {results_dir}/ensemble_results.pkl")

    # Plot ensemble ROC curve
    plot_ensemble_roc_curve(true_labels, ensemble_probs, args.n_classes, results_dir)

    # Plot ensemble PR curve
    plot_ensemble_pr_curve(true_labels, ensemble_probs, args.n_classes, results_dir)

    # Plot ensemble confusion matrix
    plot_ensemble_confusion_matrix(true_labels, ensemble_preds, args.n_classes, results_dir)

    # Plot performance comparison
    plot_performance_comparison(all_results, ensemble_results, results_dir)

    logger.info("Ensemble plots saved in the results directory.")