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
from graph_train_loop import train_graph_multi_wsi
from auxiliary_functions import seed_everything
from graph_model import KRAG_Classifier

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")



def minority_sampler(train_graph_dict):

    # calculate weights for minority oversampling
    count = []
    for k, v in train_graph_dict.items():
        count.append(v[1].item())
    counter = Counter(count)
    class_count = np.array(list(counter.values()))
    weight = 1 / class_count
    samples_weight = np.array([weight[t] for t in count])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples=len(samples_weight),  replacement=True)

    return sampler

def arg_parse():

    parser = argparse.ArgumentParser(description="self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['R4RA', 'NSCLC', 'CAMELYON16', 'Sjogren'], help="Dataset name")
    parser.add_argument("--directory", type=str, default="/data/scratch/wpw030/KRAG", help="Location of data dictionaries and results folder. Checkpoints will be kept here as well. Change to required location")
    parser.add_argument("--embedding_vector_size", type=int, default=1000, help="Embedding vector size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Size of hidden network dimension")
    parser.add_argument("--embedding_net", type=str, default="vgg16", choices=['resnet18', 'vgg16', 'convnext'], help="feature extraction network used")
    parser.add_argument("--graph_mode", type=str, default="krag", choices=['knn', 'rag', 'krag'], help="Change type of graph used for training here")
    parser.add_argument("--convolution", type=str, default="GAT", choices=['GAT', 'GCN', 'GIN', 'GraphSAGE'], help="Change type of graph convolution used")
    parser.add_argument("--attention", type=bool, default=False, help="Whether to use an attention pooling mechanism before input into classification fully connected layers")
    parser.add_argument("--positional_encoding", default=True, help="Add Random Walk positional encoding to the graph")
    parser.add_argument("--encoding_size", type=float, default=0, help="Size Random Walk positional encoding")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--pooling_ratio", type=float, default=0.7, help="Pooling ratio")
    parser.add_argument("--heads", type=int, default=2, help="Number of GAT heads")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
    parser.add_argument("--scheduler", type=str, default=1, help="learning rate schedule")
    parser.add_argument("--checkpoint", action="store_false", default=True, help="Enable checkpointing of GNN weights. Set to False if you don't want to store checkpoints.")
    parser.add_argument('--multistain', type=bool, default=False, help='Whether the dataset contains multiple types of staining.')
    parser.add_argument('--stain_type', type=str, default='all', help='Type of stain used.')
    parser.add_argument("--l1_norm", type=int, default=0.00001, help="L1-norm to regularise loss function")

    return parser.parse_args()


def main(args):

    seed_everything(args.seed)

    current_directory = args.directory
    run_results_folder = f"graph_{args.graph_mode}_{args.convolution}_PE_{args.encoding_size}_att_{args.attention}_{args.embedding_net}_{args.dataset_name}_{args.seed}_{args.heads}_{args.pooling_ratio}_{args.learning_rate}_{args.scheduler}_{args.stain_type}_L1_{args.l1_norm}"
    results = os.path.join(current_directory, "results/" + run_results_folder)
    checkpoints = results + "/checkpoints"
    os.makedirs(results, exist_ok = True)
    os.makedirs(checkpoints, exist_ok = True)


    # load pickled graphs

    if args.encoding_size == 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)

    if args.encoding_size > 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)


    # load stratified random split train/test folds
    with open(current_directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    # load stratified random split train/test folds
    #with open("/data/DERI-Krag/CAMELYON16/train_test_strat_splits_CAMELYON16.pkl", "rb") as splits:
    #    sss_folds = pickle.load(splits)


    mean_best_acc = []
    mean_best_AUC = []

    training_folds = []
    testing_folds = []
    for folds, splits in sss_folds.items():
        for i, (split, patient_ids) in enumerate(splits.items()):
            if i == 0:
                train_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
                training_folds.append(train_dict)
            if i ==1:
                test_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
                testing_folds.append(test_dict)

    for fold_idx, (train_fold, test_fold) in enumerate(zip(training_folds, testing_folds)):


        # initialising new graph, loss, optimiser between folds
        graph_net = KRAG_Classifier(args.embedding_vector_size, hidden_dim= args.hidden_dim, num_classes= args.n_classes, heads= args.heads, pooling_ratio= args.pooling_ratio, walk_length= args.encoding_size, conv_type= args.convolution, attention= args.attention)
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = optim.AdamW(graph_net.parameters(), lr=args.learning_rate, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[25, 50, 75], gamma=0.1)
        if use_gpu:
            graph_net.cuda()

        # oversampling of minority class
        sampler = minority_sampler(train_fold)

        train_graph_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler, drop_last=False)
        test_graph_loader = DataLoader(test_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        _, results_dict, best_acc, best_AUC = train_graph_multi_wsi(graph_net, train_graph_loader, test_graph_loader, loss_fn, optimizer_ft, lr_scheduler, l1_norm=args.l1_norm, n_classes=args.n_classes, num_epochs=args.num_epochs, checkpoint=args.checkpoint, checkpoint_path= checkpoints + "/checkpoint_fold_" + str(fold_idx) + "_epoch_")

        # save results to csv file
        mean_best_acc.append(best_acc.item())
        mean_best_AUC.append(best_AUC.item())

        df_results = pd.DataFrame.from_dict(results_dict)
        df_results.to_csv(results + "/" + run_results_folder + "_fold_" + str(fold_idx) + ".csv", index=False)

    average_best_acc = sum(mean_best_acc) / len(mean_best_acc)
    std_best_acc = statistics.pstdev(mean_best_acc)
    mean_best_acc.append(average_best_acc)
    mean_best_acc.append(std_best_acc)

    average_best_AUC = sum(mean_best_AUC) / len(mean_best_AUC)
    std_best_AUC = statistics.pstdev(mean_best_AUC)
    mean_best_AUC.append(average_best_AUC)
    mean_best_AUC.append(std_best_AUC)

    summary =[mean_best_acc] + [mean_best_AUC]
    summary_df = pd.DataFrame(summary, index=['val_accuracy', 'val_AUC']).transpose()
    summary_df.to_csv(results + "/" + run_results_folder + "_summary_best_scores.csv", index=0)


# %%

if __name__ == "__main__":

    datasets = ['R4RA', 'Sjogren', 'CAMELYON16', 'NSCLC']
    graph_types = ['krag']
    random_seeds = [42]
    #stains = ['HE', 'CD20', 'CD138', 'CD3', 'CD21']
    #stains = ['H&E', 'CD68', 'CD20', 'CD138']
    #stains = ['all']
    stains = ['CD20']
    for graph_type in graph_types:
        for stain in stains:
            for seed in random_seeds:
                args = arg_parse()
                args.seed = seed
                args.directory = "/data/scratch/wpw030/Sjogren_patches/results_1/"
                args.checkpoint = True
                args.dataset_name = "Sjogren"
                args.n_classes = 2
                args.embedding_net = 'vgg16'
                args.convolution = 'GAT'
                args.graph_mode = graph_type
                args.attention = False
                args.encoding_size = 20
                args.learning_rate = 0.00001
                args.scheduler = 'L2_0.01'
                args.num_epochs = 150
                args.multistain = True
                args.stain_type = stain
                args.l1_norm = 0.0
                main(args)