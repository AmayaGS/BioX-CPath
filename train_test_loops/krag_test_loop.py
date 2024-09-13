
import numpy as np
from collections import defaultdict

import torch

from utils.profiling_utils import test_profiler
from utils.model_utils import process_model_output

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
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
        'all_probs': all_probs,
        'all_labels': all_labels,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'incorrect_predictions': incorrect_predictions
    }

    return results_dict