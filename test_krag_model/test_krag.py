# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:30:22 2024

@author: AmayaGS
"""

# Misc
import os
import pickle
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# KRAG functions
from utils.auxiliary_functions import seed_everything
from models.krag_model import KRAG_Classifier
from train_test_loops.krag_train_val_loop import test_graph_multi_wsi

use_gpu = torch.cuda.is_available()


def test_krag(args):
    seed_everything(args.seed)
    current_directory = args.directory

    # Load the splits dictionary
    with open(current_directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as file:
        splits = pickle.load(file)

    # Collect results for all folds
    all_results = []
    all_metrics = {}

    for fold in splits.keys():
        print(f"Testing fold: {fold}")

        # Get test IDs for this fold
        test_ids = splits[fold]['Test']

        # Load the graph dictionary
        graph_dict_path = current_directory + f"/dictionaries/{args.graph_mode}_dict_{args.dataset_name}"
        if args.encoding_size > 0:
            graph_dict_path += f"_positional_encoding_{args.encoding_size}"
        graph_dict_path += f"_{args.embedding_net}_{args.stain_type}.pkl"

        with open(graph_dict_path, "rb") as file:
            graph_dict = pickle.load(file)

        # Initialize the model
        graph_net = KRAG_Classifier(
            in_features=args.embedding_vector_size,
            hidden_dim=args.hidden_dim,
            num_classes=args.n_classes,
            heads=args.heads,
            pooling_ratio=args.pooling_ratio,
            walk_length=args.encoding_size,
            conv_type=args.convolution
        )

        # Load the trained model weights for this fold
        checkpoint_path = os.path.join(args.checkpoint_weights, f"best_model_{fold}.pth")
        checkpoint = torch.load(checkpoint_path)
        graph_net.load_state_dict(checkpoint['model_state_dict'])

        if use_gpu:
            graph_net.cuda()

        graph_net.eval()

        # Prepare the test loader for this fold
        test_graphs = {k: graph_dict[k] for k in test_ids}
        test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False, num_workers=args.num_workers)

        # Define loss function
        loss_fn = nn.CrossEntropyLoss()

        # Perform testing
        labels, probabilities, conf_matrix, sensitivity, specificity = test_graph_multi_wsi(
            graph_net, test_loader, loss_fn, n_classes=args.n_classes
        )

        # Collect results for this fold
        fold_results = pd.DataFrame({
            'Fold': [fold] * len(test_ids),
            'Patient_ID': test_ids,
            'True_Label': labels,
            'Predicted_Probabilities': [p.tolist() for p in probabilities]
        })
        all_results.append(fold_results)

        # Collect metrics for this fold
        all_metrics[fold] = {
            'confusion_matrix': conf_matrix.tolist(),
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    # Combine results from all folds
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(current_directory + f"/test_results_{args.graph_mode}_{args.dataset_name}.csv", index=False)

    # Save combined metrics
    with open(current_directory + f"/test_metrics_{args.graph_mode}_{args.dataset_name}.pkl", "wb") as file:
        pickle.dump(all_metrics, file)

    print(f"Testing completed for all folds. Results saved.")