# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:34:24 2023

@author: AmayaGS
"""

import torch

use_gpu = torch.cuda.is_available()

import gc
gc.enable()


def slide_att_scores(graph_net, test_loader, patient_ID, loss_fn, n_classes):
    graph_net.eval()
    attention_scores= {}

    data = test_loader.dataset[0][0]
    label = test_loader.dataset[0][1]
    metadata = test_loader.dataset[0][2]
    filenames = metadata['filenames']

    with torch.no_grad():
        if use_gpu:
            data, label = data.cuda(), label.cuda()

    logits, Y_prob, all_patches_per_layer, all_patches_cumulative = graph_net(data, filenames)

    attention_scores[patient_ID] = {
        'per_layer': all_patches_per_layer,
        'cumulative': all_patches_cumulative
    }

    Y_hat = Y_prob.argmax(dim=1)

    return attention_scores, label, Y_hat


    # test_acc_logger.log(Y_hat, label)

    # test_acc += torch.sum(Y_hat == label.data)
    # test_count += 1

    # loss = loss_fn(logits, label)
    # test_loss += loss.item()

    # prob.append(Y_prob.detach().to('cpu').numpy())
    # labels.append(label.item())

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
