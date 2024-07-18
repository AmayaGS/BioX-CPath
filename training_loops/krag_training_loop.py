# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 17:34:24 2023

@author: AmayaGS
"""

import time
import os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

import torch
from torch_geometric.data import Data

from utils.auxiliary_functions import Accuracy_Logger, setup_logger

use_gpu = torch.cuda.is_available()

import gc
gc.enable()


def l1_regularization(model, l1_norm):
    weights = sum(torch.abs(p).sum() for p in model.parameters())
    return weights * l1_norm


def randomly_shuffle_graph(data, seed=None):
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Randomly shuffle the node features
    shuffled_features = data.x[torch.randperm(data.num_nodes)]
    shuffled_rw = data.random_walk_pe[torch.randperm(data.num_nodes)]

    # Randomly shuffle the edge index
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    shuffled_edge_index = edge_index[:, torch.randperm(num_edges)]

    # Create a new Data object with the shuffled node features and edge index
    shuffled_data = Data(
        x= shuffled_features,
        edge_index= shuffled_edge_index,
        random_walk_pe= shuffled_rw
    )

    return shuffled_data


def train_epoch(graph_net, train_loader, loss_fn, optimizer):

    train_loss = 0
    train_correct = 0
    train_total = 0

    graph_net.train()

    for patient_ID, graph_object in train_loader.dataset.items():
        data, label, _, _ = graph_object
        if use_gpu:
            data, label = data.cuda(), label.cuda()

        logits, Y_prob = graph_net(data)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = Y_prob.argmax(dim=1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        # Explicit deletion of variables
        del graph_object, data, label, logits, Y_prob, predicted, loss
        # Clear CUDA cache if using GPU
        if use_gpu:
            torch.cuda.empty_cache()
        # Garbage collection
        gc.collect()

    return train_loss / len(train_loader.dataset), train_correct / train_total


def evaluate_model(graph_net, val_loader, loss_fn, n_classes):

    val_loss = 0
    val_correct = 0
    val_total = 0
    all_probs = []
    all_labels = []

    graph_net.eval()

    with torch.no_grad():
        for patient_ID, graph_object in val_loader.dataset.items():
            data, label, _, _ = graph_object
            if use_gpu:
                data, label = data.cuda(), label.cuda()

            logits, Y_prob = graph_net(data)
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            _, predicted = torch.max(Y_prob, 1)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            del graph_object, data, label, logits, Y_prob, predicted, loss
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


def train_graph(graph_net, train_loader, val_loader, loss_fn, optimizer, logging_file_path, n_classes, num_epochs, checkpoint, checkpoint_path="PATH_checkpoints"):

    since = time.time()
    best_val_acc = 0.
    best_val_AUC = 0.

    results_dict = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': []
    }

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(graph_net, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy, val_auc, conf_matrix, class_report = evaluate_model(graph_net, val_loader, loss_fn, n_classes)
        logger = setup_logger(logging_file_path)

        results_dict['train_loss'].append(train_loss)
        results_dict['train_accuracy'].append(train_accuracy)
        results_dict['val_loss'].append(val_loss)
        results_dict['val_accuracy'].append(val_accuracy)
        results_dict['val_auc'].append(val_auc)

        # print(f"Epoch {epoch+1}/{num_epochs}")
        # print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        # print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # print("Classification Report:")
        # print(class_report)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation AUC: {val_auc:.4f}")
        logger.info(f"{conf_matrix}")
        logger.info(f"{class_report}")

        if val_accuracy >= best_val_acc:
            best_val_acc = val_accuracy

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(graph_net.state_dict(), checkpoint_weights)

        if val_auc >= best_val_AUC:
            best_val_AUC = val_auc

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(graph_net.state_dict(), checkpoint_weights)

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return graph_net, results_dict, best_val_acc, best_val_AUC


def train_graph_multi_wsi(graph_net, train_loader, test_loader, loss_fn, optimizer, lr_scheduler, l1_norm, n_classes, num_epochs, checkpoint, checkpoint_path="PATH_checkpoints"):


    since = time.time()
    best_acc = 0.
    best_AUC = 0.

    results_dict = {}

    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []

    sensitivity_list = []
    specificity_list = []

    for epoch in range(num_epochs):

        ##################################
        # TRAIN
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        train_loss = 0
        train_acc = 0
        train_count = 0
        graph_net.train()

        print("Epoch {}/{}".format(epoch, num_epochs), flush=True)
        print('-' * 10)

        for batch_idx, (patient_ID, graph_object) in enumerate(train_loader.dataset.items()):
            #print(patient_ID)

            #if patient_ID != "QMUL-R4RA-X990":

            data, label, _, _ = graph_object

            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

            #shuffled_data = randomly_shuffle_graph(data, seed=42)
            #logits, Y_prob = graph_net(shuffled_data)

            logits, Y_prob = graph_net(data)

            Y_hat = Y_prob.argmax(dim=1)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            train_acc += torch.sum(Y_hat == label.data)
            train_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            del data, graph_object, logits, Y_prob, Y_hat
            gc.collect()

            #loss = loss_fn(logits, label)

            #train_acc += torch.sum(Y_hat == label.data)
            #train_count += 1

            #optimizer.zero_grad()
            #l1_loss = l1_regularization(graph_net, l1_norm)
            #reg_loss = loss + l1_loss
            #reg_loss.backward()
            #optimizer.step()

            #train_loss += reg_loss.item()

            #del data, graph_object, logits, Y_prob, Y_hat
            #gc.collect()

        #lr_scheduler.step()

        total_loss = train_loss / train_count
        train_accuracy =  train_acc / train_count

        train_loss_list.append(total_loss)
        train_accuracy_list.append(train_accuracy.item())

        print()
        print('Epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(epoch, total_loss, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        ################################
        # TEST/EVAL
        graph_net.eval()

        val_acc_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_acc = 0
        val_count = 0

        prob = []
        labels = []

        for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):
            #if patient_ID != "QMUL-R4RA-X990":

            data, label, _, _ = graph_object

            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

            #shuffled_data = randomly_shuffle_graph(data, seed=42)
            #logits, Y_prob = graph_net(shuffled_data)

            logits, Y_prob = graph_net(data)
            Y_hat = Y_prob.argmax(dim=1)
            val_acc_logger.log(Y_hat, label)

            val_acc += torch.sum(Y_hat == label.data)
            val_count += 1

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            prob.append(Y_prob.detach().to('cpu').numpy())
            labels.append(label.item())

            del data, graph_object, logits, Y_prob, Y_hat
            gc.collect()

            #loss = loss_fn(logits, label)
            #l1_loss = l1_regularization(graph_net, l1_norm)
            #reg_loss = loss + l1_loss
            #val_loss += reg_loss.item()

            #prob.append(Y_prob.detach().to('cpu').numpy())
            #labels.append(label.item())

            #del data, graph_object, logits, Y_prob, Y_hat
            #gc.collect()

        val_loss /= val_count
        val_accuracy = val_acc / val_count

        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy.item())

        if n_classes == 2:
            prob = np.stack(prob, axis=1)[0]
            val_auc = roc_auc_score(labels, prob[:, 1])
        else:
            binary_labels = label_binarize(labels, classes=range(n_classes))
            prob = np.stack(prob, axis=0)

            aucs = []
            for i in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, i], prob[:, i])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            val_auc = np.nanmean(np.array(aucs))

        val_auc_list.append(val_auc)

        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

        print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_auc, val_accuracy), flush=True)

        print(conf_matrix)

        if n_classes == 2:
            sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
            specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print('Sensitivity: ', sensitivity)
            print('Specificity: ', specificity)

            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)

        if val_accuracy >= best_acc:
            best_acc = val_accuracy

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(graph_net.state_dict(), checkpoint_weights)

        if val_auc >= best_AUC:
            best_AUC = val_auc

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(graph_net.state_dict(), checkpoint_weights)

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    #if checkpoint:
    #    graph_net.load_state_dict(torch.load(checkpoint_weights), strict=True)

    if n_classes == 2:
        results_dict = {'train_loss': train_loss_list,
                        'val_loss': val_loss_list,
                        'train_accuracy': train_accuracy_list,
                        'val_accuracy': val_accuracy_list,
                        'val_auc': val_auc_list,
                        'sensitivity': sensitivity_list,
                        'specificity': specificity_list
                        }

    else:
        results_dict = {'train_loss': train_loss_list,
                        'val_loss': val_loss_list,
                        'train_accuracy': train_accuracy_list,
                        'val_accuracy': val_accuracy_list,
                        'val_auc': val_auc_list
                        }


    return graph_net, results_dict, best_acc, best_AUC


# TEST

def test_graph_multi_wsi(graph_net, test_loader, loss_fn, n_classes=2):

    since = time.time()

    test_acc_logger = Accuracy_Logger(n_classes)
    test_loss = 0.
    test_acc = 0
    test_count = 0

    prob = []
    labels = []
    test_loss_list = []
    test_accuracy_list = []
    test_auc_list = []

    graph_net.eval()

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        data, label = graph_object

        with torch.no_grad():
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

        logits, Y_prob = graph_net(data)
        #logits, Y_prob, x, edge_index, perm, score = graph_net(data)
        Y_hat = Y_prob.argmax(dim=1)
        test_acc_logger.log(Y_hat, label)

        test_acc += torch.sum(Y_hat == label.data)
        test_count += 1

        loss = loss_fn(logits, label)
        test_loss += loss.item()

        prob.append(Y_prob.detach().to('cpu').numpy())
        labels.append(label.item())

        #data = Data(x=x, edge_index=edge_index)
        #graph = to_networkx(data) # convert torch geometric Data to Networkx graph object

        #for i, node in enumerate(graph.nodes(data=True)): # assign attention score as node attribute to each node
        #    node[1]['score'] = float(score[i])

        #node_colors = [data['score'] for _, data in graph.nodes(data=True)] # assign colors to the nodes based on the node attribute

        #plt.figure(figsize=(15, 10))
        # Use the spring layout for better node placement
        #pos = nx.spring_layout(graph)
        #pos = nx.spectral_layout(graph)
        #pos = nx.kamada_kawai_layout(graph)
        #pos = nx.spring_layout(graph, seed=42, iterations=200)

        # Draw nodes with edgecolor set to 'none'
        #nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.coolwarm, node_size=50, alpha=0.8)

        # Draw edges
        #nx.draw_networkx_edges(graph, pos, width=0.1, alpha=0.8)

        # Display the graph without node labels
        #plt.title('Graph with Node Colors Based on all Scores')
        #plt.axis('off')  # Turn off axis labels
        #plt.show()

        del data, logits, Y_prob, Y_hat
        gc.collect()

    test_loss /= test_count
    test_accuracy = test_acc / test_count

    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy.item())

    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
        test_auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        prob =  np.stack(prob, axis=1)[0]
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        test_auc = np.nanmean(np.array(aucs))

    test_auc_list.append(test_auc)

    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

    print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_auc, test_accuracy), flush=True)

    print(conf_matrix)

    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        print('Sensitivity: ', sensitivity)
        print('Specificity: ', specificity)

    elapsed_time = time.time() - since

    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return labels, prob, conf_matrix, sensitivity, specificity