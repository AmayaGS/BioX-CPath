


import numpy as np
from collections import defaultdict

import torch

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

from utils.auxiliary_functions import setup_logger

from utils.plotting_functions import plot_roc_curve

use_gpu = torch.cuda.is_available()


def test_graph(graph_net, test_loader, loss_fn, n_classes, logging_file_path, fold):
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_probs = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    incorrect_predictions = []

    results_dict = {}

    logger = setup_logger(logging_file_path)

    graph_net.eval()
    with torch.no_grad():
        for patient_ID, graph_object in test_loader.dataset.items():
            data, label, _, _ = graph_object
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            logits, Y_prob = graph_net(data)
            loss = loss_fn(logits, label)

            test_loss += loss.item()
            _, predicted = torch.max(Y_prob, 1)
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
        'all_probs': all_probs,
        'all_labels': all_labels,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'incorrect_predictions': incorrect_predictions
    }

    return results_dict

# Usage in main script:
# from krag_testing_loop import test_model
#
# test_results = test_model(graph_net, test_loader, loss_fn, n_classes, use_gpu, logger)