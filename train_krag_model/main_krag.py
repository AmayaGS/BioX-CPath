# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:43:55 2024

@author: AmayaGS
"""

# Misc
import os
import os.path
import numpy as np
import pandas as pd
import statistics
from collections import Counter
import pickle

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# PyG
from torch_geometric.loader import DataLoader

# KRAG functions
from training_loops.krag_training_loop import train_graph_multi_wsi
from utils.auxiliary_functions import seed_everything
from models.krag_model import KRAG_Classifier

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()

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


def train_krag(args):

    seed_everything(args.seed)

    current_directory = args.directory
    run_results_folder = f"graph_{args.graph_mode}_{args.convolution}_PE_{args.encoding_size}_{args.embedding_net}_{args.dataset_name}_{args.seed}_{args.heads}_{args.pooling_ratio}_{args.learning_rate}_{args.scheduler}_{args.stain_type}_L1_{args.l1_norm}"
    results = os.path.join(current_directory, "output/" + run_results_folder)
    checkpoints = results + "/checkpoints"
    os.makedirs(results, exist_ok = True)
    os.makedirs(checkpoints, exist_ok = True)

    # load pickled graphs
    if args.encoding_size == 0:
        with open(current_directory + f"/dictionaries/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)

    if args.encoding_size > 0:
        with open(current_directory + f"/dictionaries/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)


    # load stratified random split train/test folds
    with open(current_directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    mean_best_acc = []
    mean_best_AUC = []

    training_folds = []
    validation_folds = []
    testing_folds = []
    for folds, splits in sss_folds.items():
        for i, (split, patient_ids) in enumerate(splits.items()):
            if i == 0:
                train_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
                training_folds.append(train_dict)
            if i== 1:
                val_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
                validation_folds.append(val_dict)
            if i == 2:
                test_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
                testing_folds.append(test_dict)

    for fold_idx, (train_fold, val_folds) in enumerate(zip(training_folds, validation_folds)):

        # initialising new graph, loss, optimiser between folds
        graph_net = KRAG_Classifier(args.embedding_vector_size, hidden_dim= args.hidden_dim, num_classes= args.n_classes, heads= args.heads, pooling_ratio= args.pooling_ratio, walk_length= args.encoding_size, conv_type= args.convolution)
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = optim.AdamW(graph_net.parameters(), lr=args.learning_rate, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[25, 50, 75], gamma=0.1)
        if use_gpu:
            graph_net.cuda()

        # oversampling of minority class
        sampler = minority_sampler(train_fold)

        train_graph_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler, drop_last=False)
        val_graph_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        #test_graph_loader = DataLoader(test_fold, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        _, results_dict, best_acc, best_AUC = train_graph_multi_wsi(graph_net, train_graph_loader, val_graph_loader, loss_fn, optimizer_ft, lr_scheduler, l1_norm=args.l1_norm, n_classes=args.n_classes, num_epochs=args.num_epochs, checkpoint=args.checkpoint, checkpoint_path= checkpoints + "/checkpoint_fold_" + str(fold_idx) + "_epoch_")

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
