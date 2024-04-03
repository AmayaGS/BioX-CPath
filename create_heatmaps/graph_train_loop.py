# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:34:24 2023

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch

from auxiliary_functions import Accuracy_Logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc
gc.enable()



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

    attention_scores= {}

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        data, label, files, filenames = graph_object

        with torch.no_grad():
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

        logits, Y_prob, all_patches = graph_net(data, filenames)

        attention_scores[patient_ID] = [all_patches]

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

    return test_accuracy, test_auc, conf_matrix, sensitivity, specificity, attention_scores