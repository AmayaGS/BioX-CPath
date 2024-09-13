# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 17:34:24 2023

@author: AmayaGS
"""

import time
import os.path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

import torch

from utils.profiling_utils import train_profiler, test_profiler
from utils.plotting_functions_utils import plot_training_results

import gc
gc.enable()
use_gpu = torch.cuda.is_available()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train_val_loop(model, train_loader, val_loader, loss_fn, optimizer, results_dir, logger, fold, n_classes, num_epochs,
                   checkpoint, checkpoint_path):
    since = time.time()
    best_val_acc = 0.
    best_val_AUC = 0.

    best_model_weights = os.path.join(checkpoint_path, "best_val_models")
    os.makedirs(best_model_weights, exist_ok=True)

    results_dict = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': []
    }

    for epoch in range(num_epochs):
        train_profiler.update_peak_memory()

        epoch_start = time.time()
        train_loss, train_accuracy, class_correct, class_total = train_loop(model,
                                                                            train_loader,
                                                                            loss_fn,
                                                                            optimizer,
                                                                            n_classes)
        val_loss, val_accuracy, val_auc, conf_matrix, class_report = eval_loop(model,
                                                                               val_loader,
                                                                               loss_fn,
                                                                               n_classes)
        epoch_time = time.time() - epoch_start
        train_profiler.update_epoch_time(epoch_time)

        results_dict['train_loss'].append(train_loss)
        results_dict['train_accuracy'].append(train_accuracy)
        results_dict['val_loss'].append(val_loss)
        results_dict['val_accuracy'].append(val_accuracy)
        results_dict['val_auc'].append(val_auc)

        logger.info(f"\n{'=' * 25} Split {fold} {'=' * 25}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        for i in range(n_classes):
            logger.info(f"Class {i}: {class_correct[i]}/{class_total[i]}")
        logger.info(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
        logger.info(f"\n{conf_matrix}")
        logger.info(f"\n{class_report}")

        if val_accuracy >= best_val_acc:
            best_val_acc = val_accuracy

            if checkpoint:
                checkpoint_weights = checkpoint_path + "/checkpoint_fold_" + str(fold) + "_epoch_" + str(epoch) + ".pth"
                torch.save(model.state_dict(), checkpoint_weights)
                torch.save(model.state_dict(), best_model_weights + f"/checkpoint_fold_{fold}_accuracy.pth")


        if val_auc >= best_val_AUC:
            best_val_AUC = val_auc

            if checkpoint:
                checkpoint_weights = checkpoint_path + "/checkpoint_fold_" + str(fold) + "_epoch_" + str(epoch) + ".pth"
                torch.save(model.state_dict(), checkpoint_weights)
                torch.save(model.state_dict(), best_model_weights + f"/checkpoint_fold_{fold}_auc.pth")

    plot_training_results(results_dict, fold, results_dir)

    elapsed_time = time.time() - since
    print()
    logger.info("Training & validation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return model, results_dict, best_val_acc, best_val_AUC

def train_loop(model, train_loader, loss_fn, optimizer, n_classes):
    train_loss = 0
    train_correct = 0
    train_total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    model.train()

    for patient_ID, data_object in train_loader.dataset.items():
        train_profiler.update_peak_memory()
        data, label, _ = data_object
        if use_gpu:
            data, label = data.cuda(), label.cuda()

        logits, Y_prob, predicted, loss = process_model_output(model(data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        # Count correct predictions for each class
        for i in range(n_classes):
            class_correct[i] += ((predicted == label) & (label == i)).sum().item()
            class_total[i] += (label == i).sum().item()

        # Explicit deletion of variables
        del data_object, data, label, logits, Y_prob, predicted, loss
        # Clear CUDA cache if using GPU
        if use_gpu:
            torch.cuda.empty_cache()
        # Garbage collection
        gc.collect()

    return train_loss / len(train_loader.dataset), train_correct / train_total, class_correct, class_total


def eval_loop(model, val_loader, loss_fn, n_classes):
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_probs = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for patient_ID, data_object in val_loader.dataset.items():
            train_profiler.update_peak_memory()
            data, label, _ = data_object
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            logits, Y_prob, predicted, loss = process_model_output(model(data))
            val_loss += loss.item()

            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            del data_object, data, label, logits, Y_prob, predicted, loss
            # Clear CUDA cache if using GPU
            if use_gpu:
                torch.cuda.empty_cache()
            # Garbage collection
            gc.collect()

    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / val_total

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels)

    if n_classes == 2:
        val_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=range(n_classes))
        val_auc = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')

    predicted_labels = np.argmax(all_probs, axis=1)
    conf_matrix = confusion_matrix(all_labels, np.argmax(all_probs, axis=1))
    class_report = classification_report(all_labels, predicted_labels, zero_division=0)

    return val_loss, val_accuracy, val_auc, conf_matrix, class_report


def test_loop(model, test_loader, loss_fn, n_classes, logger, fold):
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

            logits, Y_prob, predicted, loss = process_model_output(model(data))

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



