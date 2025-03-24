import os
import time
from collections import defaultdict

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score

import torch

from utils.profiling_utils import train_profiler
from utils.model_utils import process_model_output
from utils.plotting_functions_utils import plot_training_results

import gc
gc.enable()
use_gpu = torch.cuda.is_available()

def train_val_loop(args, model, train_loader, val_loader, loss_fn, optimizer, results_dir, logger, fold, n_classes,
                   num_epochs, checkpoint, checkpoint_path):
    since = time.time()
    best_val_acc = 0.
    best_val_AUC = 0.
    best_val_loss = float('inf')
    best_val_acc_loss = float('inf')
    best_val_auc_loss = float('inf')

    patience = 15
    patience_counter = 0
    improvement_threshold = 0.001

    best_model_weights = os.path.join(checkpoint_path, "best_val_models")
    os.makedirs(best_model_weights, exist_ok=True)

    results_dict = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_precision': []
    }

    for epoch in range(num_epochs):
        train_profiler.update_peak_memory()

        epoch_start = time.time()
        train_loss, train_accuracy, class_correct, class_total = train_loop(args,
                                                                            model,
                                                                            train_loader,
                                                                            loss_fn,
                                                                            optimizer,
                                                                            n_classes)
        val_loss, val_accuracy, val_auc, val_pr, conf_matrix, class_report = eval_loop(args,
                                                                               model,
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
        results_dict['val_precision'].append(val_pr)

        logger.info(f"\n{'=' * 25} Split {fold} {'=' * 25}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        for i in range(n_classes):
            logger.info(f"Class {i}: {class_correct[i]}/{class_total[i]}")
        logger.info(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
        logger.info(f"\n{conf_matrix}")
        logger.info(f"\n{class_report}")

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch + 1
            improved = True
            if checkpoint:
                torch.save(model.state_dict(),
                           os.path.join(best_model_weights, f"checkpoint_fold_{fold}_loss.pth"))

        if val_accuracy > best_val_acc or (val_accuracy == best_val_acc and val_loss < best_val_acc_loss):
            best_val_acc = val_accuracy
            best_val_acc_loss = val_loss  # Keep track of corresponding loss
            best_acc_epoch = epoch + 1
            improved = True
            if checkpoint:
                torch.save(model.state_dict(),
                           os.path.join(best_model_weights, f"checkpoint_fold_{fold}_accuracy.pth"))

        if val_auc > best_val_AUC or (val_auc == best_val_AUC and val_loss < best_val_auc_loss):
            best_val_AUC = val_auc
            best_val_auc_loss = val_loss  # Keep track of corresponding loss
            best_auc_epoch = epoch + 1
            improved = True
            if checkpoint:
                torch.save(model.state_dict(),
                           os.path.join(best_model_weights, f"checkpoint_fold_{fold}_auc.pth"))

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            logger.info(f"Best model checkpoints:")
            logger.info(f"  Loss checkpoint: epoch {best_loss_epoch}, val_loss = {best_val_loss:.4f}")
            logger.info(f"  Accuracy checkpoint: epoch {best_acc_epoch}, val_acc = {best_val_acc:.4f}, val_loss = {best_val_acc_loss:.4f}")
            logger.info(f"  AUC checkpoint: epoch {best_auc_epoch}, val_auc = {best_val_AUC:.4f}, val_loss = {best_val_auc_loss:.4f}")
            break

    plot_training_results(results_dict, fold, results_dir)

    elapsed_time = time.time() - since
    print()
    logger.info("Training & validation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return model, results_dict, best_val_acc, best_val_AUC

def train_loop(args, model, train_loader, loss_fn, optimizer, n_classes):
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

        logits, Y_prob, predicted, loss = process_model_output(args, model(data, label), loss_fn)
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


def eval_loop(args, model, val_loader, loss_fn, n_classes):
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

            logits, Y_prob, predicted, loss = process_model_output(args, model(data, label), loss_fn)
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
        val_precision = average_precision_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=range(n_classes))
        val_auc = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')
        all_preds = np.argmax(all_probs, axis=1)
        val_precision = average_precision_score(all_labels,
                                                label_binarize(all_preds, classes=range(args.n_classes)),
                                                average='macro')

    predicted_labels = np.argmax(all_probs, axis=1)
    conf_matrix = confusion_matrix(all_labels, np.argmax(all_probs, axis=1))
    class_report = classification_report(all_labels, predicted_labels, zero_division=0)

    return val_loss, val_accuracy, val_auc, val_precision, conf_matrix, class_report

