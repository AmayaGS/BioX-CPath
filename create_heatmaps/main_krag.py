# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:43:55 2024

@author: AmayaGS
"""

# -*- coding: utf-8 -*-

"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

# Misc
import os
import os.path
from pathlib import Path
import numpy as np
import pandas as pd
import statistics
from collections import Counter
import pickle
import argparse

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# PyG
from torch_geometric.loader import DataLoader

# KRAG functions
from graph_train_loop import test_graph_multi_wsi
from auxiliary_functions import seed_everything
from graph_model import KRAG_Classifier

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

def arg_parse():

    parser = argparse.ArgumentParser(description="self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'LUAD', 'LSCC'], help="Dataset name")
    parser.add_argument("--directory", type=str, default="/data/scratch/wpw030/KRAG", help="Location of data dictionaries and results folder. Checkpoints will be kept here as well. Change to required location")
    parser.add_argument("--embedding_vector_size", type=int, default=1000, help="Embedding vector size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Size of hidden network dimension")
    parser.add_argument("--embedding_net", type=str, default="vgg16", choices=['resnet18', 'vgg16', 'convnext'], help="feature extraction network used")
    parser.add_argument("--graph_mode", type=str, default="krag", choices=['knn', 'rag', 'krag'], help="Change type of graph used for training here")
    parser.add_argument("--convolution", type=str, default="GAT", choices=['GAT', 'GCN', 'GIN', 'GraphSAGE'], help="Change type of graph convolution used")
    parser.add_argument("--attention", type=bool, default=False, help="Whether to use an attention pooling mechanism before input into classification fully connected layers")
    parser.add_argument("--positional_encoding", default=True, help="Add Random Walk positional encoding to the graph")
    parser.add_argument("--encoding_size", type=float, default=0, help="Size Random Walk positional encoding")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--pooling_ratio", type=float, default=0.7, help="Pooling ratio")
    parser.add_argument("--heads", type=int, default=2, help="Number of GAT heads")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
    parser.add_argument("--weights", type=str, default="/data/scratch/wpw030/KRAG/", help="Location of trained model weights.")
    parser.add_argument("--fold", type=str, default="0", help="Location of trained model weights.")
    return parser.parse_args()


def main(args):


    seed_everything(args.seed)

    current_directory = args.directory
    weights_directory = args.weights

    # load pickled graphs
    if args.encoding_size == 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}.pkl", "rb") as file:
            graph_dict = pickle.load(file)

    if args.encoding_size > 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}.pkl", "rb") as file:
            graph_dict = pickle.load(file)


    # load stratified random split train/test folds
    with open(current_directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    #training_folds = []
    testing_folds = []
    for fold in range(len(sss_folds)):
        test_ids = sss_folds[f'Fold {fold}']['Test']
        for patient_ids in test_ids:
            test_dict = dict(filter(lambda i:i[0] in test_ids, graph_dict.items()))
            testing_folds.append(test_dict)

    for test_fold in testing_folds:

        # initialising new graph, loss, optimiser between folds
        graph_net = KRAG_Classifier(args.embedding_vector_size, hidden_dim= args.hidden_dim, num_classes= args.n_classes, heads= args.heads, pooling_ratio= args.pooling_ratio, walk_length= args.encoding_size, conv_type= args.convolution, attention= args.attention)
        loss_fn = nn.CrossEntropyLoss()

        fold_weight = f"checkpoint_fold_{args.fold}_{args.dataset_name}.pth"
        weight_path = os.path.join(weights_directory, fold_weight)
        checkpoint = torch.load(weight_path)
        graph_net.load_state_dict(checkpoint, strict=False)

        if use_gpu:
            graph_net.cuda()

        test_graph_loader = DataLoader(test_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        test_accuracy, test_auc, conf_matrix, sensitivity, specificity, attention_scores = test_graph_multi_wsi(graph_net, test_graph_loader, loss_fn, n_classes=args.n_classes)

        print(test_accuracy, test_auc, sensitivity, specificity, conf_matrix)

        with open(current_directory + f"/attn_score_dict_fold_{args.fold}_{args.dataset_name}.pkl", "wb") as file:
            pickle.dump(attention_scores, file)


# %%

if __name__ == "__main__":
    args = arg_parse()
    args.directory = r"C:\Users\Amaya\Documents\PhD\MUSTANGv2"
    args.dataset_name = "RA"
    args.embedding_net = 'vgg16'
    args.convolution = 'GAT'
    args.graph_mode = 'krag'
    args.attention = True
    args.encoding_size = 20
    args.weights = r"C:\Users\Amaya\Documents\PhD\MUSTANGv2"
    args.fold = 2
    main(args)