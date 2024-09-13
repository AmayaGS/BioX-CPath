import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from utils.setup_utils import get_model_config
from utils.plotting_functions_utils import plot_averaged_results
from utils.plotting_functions_utils import plot_average_roc_curve, plot_average_pr_curve

from models.KRAG_model import KRAG_Classifier
from models.patchGCN_model import PatchGCN
from models.DeepGraphConv_model import DeepGraphConv
from models.GTP_model import GTP_Classifier
from models.TransMIL_model import TransMIL
from models.CLAM_model import GatedAttention as CLAM



def process_model_output(args, output, loss_fn):
    if args.model_name == 'KRAG':
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'CLAM':
        bag_weight = 0.7
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        instance_loss = results_dir['instance_loss']
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss
        return logits, Y_prob, Y_hat, total_loss

    elif args.model_name == 'PatchGCN':
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'DeepGraphConv':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'TransMIL':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'GTP':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        total_loss = loss + results_dir['mc1'] + results_dir['o1']
        return logits, Y_prob, Y_hat, total_loss

    elif args.model_name == 'HEAT':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'CAMIL':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    else:

        raise ValueError(f"Unsupported model: {args.model_name}")

def load_data(args, results_dir):
    config = get_model_config(args)
    run_settings = results_dir.split('\\')[-1]
    checkpoints = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok = True)

    graph_dict_path = args.directory + f"/dictionaries/{config['graph_mode']}_dict_{args.dataset_name}"

    if config['encoding_size'] > 0 and config['graph_mode'] == 'krag':
        graph_dict_path += f"_positional_encoding_{config['encoding_size']}"

    graph_dict_path += f"_{args.embedding_net}_{args.stain_type}.pkl"

    with open(graph_dict_path, "rb") as file:
        graph_dict = pickle.load(file)

    # load stratified random split train/test folds
    with open(args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    return run_settings, checkpoints, graph_dict, sss_folds

def prepare_data_loaders(data_dict, sss_folds):
    training_folds = []
    validation_folds = []
    testing_folds = []

    for fold, splits in sss_folds.items():
        train_dict = {k: data_dict[k] for k in splits['Train']}
        val_dict = {k: data_dict[k] for k in splits['Val']}
        test_dict = {k: data_dict[k] for k in splits['Test']}
        training_folds.append(train_dict)
        validation_folds.append(val_dict)
        testing_folds.append(test_dict)

    return training_folds, validation_folds, testing_folds

def initialise_model(args):
    if args.model_name == 'KRAG':
        model = KRAG_Classifier(args.embedding_vector_size,
                                hidden_dim=args.hidden_dim,
                                num_classes=args.n_classes,
                                heads=args.heads,
                                pooling_ratio=args.pooling_ratio,
                                walk_length=args.encoding_size,
                                conv_type=args.convolution)
    elif args.model_name == 'CLAM':
        model = CLAM(args.embedding_vector_size)
    elif args.model_name == 'DeepGraphConv':
        model = DeepGraphConv(num_features=args.embedding_vector_size,
                              hidden_dim=args.hidden_dim,
                              n_classes=args.n_classes)
    elif args.model_name == 'PatchGCN':
        model = PatchGCN(num_features=args.embedding_vector_size,
                         hidden_dim=args.hidden_dim,
                         n_classes=args.n_classes)
    elif args.model_name == 'DeepGraphConv':
        model = DeepGraphConv(num_features=args.embedding_vector_size,
                     hidden_dim=args.hidden_dim,
                     n_classes=args.n_classes)
    elif args.model_name == 'GTP':
        model = GTP_Classifier(n_class=args.n_classes,
                               n_features=args.embedding_vector_size)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.n_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()

    return model, loss_fn, optimizer, lr_scheduler

def summarise_train_results(all_results, mean_best_acc, mean_best_AUC, results_dir, run_settings):
    plot_averaged_results(all_results, results_dir + "/")

    average_best_acc = np.mean(mean_best_acc)
    std_best_acc = np.std(mean_best_acc)

    average_best_AUC = np.mean(mean_best_AUC)
    std_best_AUC = np.std(mean_best_AUC)

    summary_df = pd.DataFrame({
        'val_accuracy': [average_best_acc, std_best_acc],
        'val_AUC': [average_best_AUC, std_best_AUC]
    }, index=['mean', 'std']).T

    summary_path = f"{results_dir}/{run_settings}_train_summary_scores.csv"
    summary_df.to_csv(summary_path, index=True)

def summarise_test_results(all_results, results_dir, logger, args):
    accuracies = [r['test_accuracy'] for r in all_results]
    aucs = [r['test_auc'] for r in all_results]

    # Calculate averages and standard deviations
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Create summary dataframe
    summary_df = pd.DataFrame({
        'test_accuracy': [avg_accuracy, std_accuracy],
        'test_AUC': [avg_auc, std_auc]
    }, index=['mean', 'std']).T

    # Save summary to CSV
    run_settings = results_dir.split('\\')[-1]
    summary_path = f"{results_dir}/{run_settings}_test_summary_scores.csv"
    summary_df.to_csv(summary_path, index=True)

    logger.info(f"Average Test Accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    logger.info(f"Average Test AUC: {avg_auc:.4f} +/- {std_auc:.4f}")

    # Plot average curves
    plot_average_roc_curve(all_results, args.n_classes, results_dir)
    plot_average_pr_curve(all_results, args.n_classes, results_dir)


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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                             num_samples=len(samples_weight), replacement=True)

    return sampler

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
        x=shuffled_features,
        edge_index=shuffled_edge_index,
        random_walk_pe=shuffled_rw
    )

    return shuffled_data