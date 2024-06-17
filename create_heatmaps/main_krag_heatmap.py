# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:43:55 2024

@author: AmayaGS

"""

# Misc
import os
import os.path
import pandas as pd
import pickle
import argparse

# PyTorch
import torch
import torch.nn as nn

# PyG
from torch_geometric.loader import DataLoader

# KRAG functions
from graph_train_loop_single_image import slide_att_scores, slide_att_scores_per_layer
from auxiliary_functions import seed_everything
from graph_model import KRAG_Classifier, KRAG_Classifier_per_layer
from heatmap_generation import create_heatmaps, create_heatmaps_per_layer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")


def normalize_attention_scores_per_layer(attention_scores_dict):

    normalized_scores_dict = {}

    for patient, attention_scores_list in attention_scores_dict.items():
        # Extract all the scores from the four lists
        scores = []
        for attention_scores in attention_scores_list:
            scores.extend([score for _, score in attention_scores])

        # Find the overall minimum and maximum scores
        min_score = min(scores)
        max_score = max(scores)

        # Normalize each score using min-max normalization
        normalized_scores_list = []
        for attention_scores in attention_scores_list:
            normalized_scores = []
            for patch_name, score in attention_scores:
                if max_score - min_score == 0:
                    normalized_score = 0.5
                else:
                    normalized_score = (score - min_score) / (max_score - min_score) + 0.5
                normalized_scores.append((patch_name, normalized_score))
            normalized_scores_list.append(normalized_scores)

        normalized_scores_dict[patient] = normalized_scores_list

    return normalized_scores_dict

def arg_parse():

    parser = argparse.ArgumentParser(description="self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'LUAD', 'LSCC'], help="Dataset name")
    parser.add_argument("--directory", type=str, default="/data/scratch/wpw030/KRAG", help="Location of data dictionaries and results folder. Checkpoints will be kept here as well. Change to required location")
    parser.add_argument("--path_to_patches", type=str, default="/data/scratch/wpw030/KRAG/results/patches/", help="Location of patches")
    parser.add_argument("--heatmap_path", type=str, default="/data/scratch/wpw030/KRAG/results/heatmaps/", help="Location of saved heatmap figs")
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
    parser.add_argument("--patch_size", type=int, default=224, help="HxW size of the input patches")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
    parser.add_argument("--checkpoint_weights", type=str, default="/data/scratch/wpw030/KRAG/", help="Location of trained model weights.")
    parser.add_argument("--test_fold", type=str, default="Fold_9", help="test fold")
    parser.add_argument("--test_ids", type=str, default="Fold_9", help="specific id to create heatmap on.")
    parser.add_argument("--slide_name", type=str, default="/data/scratch/wpw030/KRAG/slide1", help="Location of slide which to create heatmap for.")
    parser.add_argument("--stain_type", type=str, default="all", help="stain type used in the image")
    parser.add_argument("--per_layer", type=bool, default=True, action='store_true')


    return parser.parse_args()


def main(args):

    seed_everything(args.seed)
    current_directory = args.directory
    os.makedirs(args.heatmap_path, exist_ok=True)
    img_folders = os.listdir(args.path_to_patches)
    heatmaps = os.listdir(args.heatmap_path)
    heatmap_list = [h[::-1].split("_", 1)[1][::-1] for h in heatmaps]


    # initialising graph, loss, optimiser between folds
    loss_fn = nn.CrossEntropyLoss()
    if per_layer:
        graph_net = KRAG_Classifier_per_layer(args.embedding_vector_size, hidden_dim= args.hidden_dim, num_classes= args.n_classes, heads= args.heads, pooling_ratio= args.pooling_ratio, walk_length= args.encoding_size, conv_type= args.convolution, attention= args.attention)
    else:
        graph_net = KRAG_Classifier(args.embedding_vector_size, hidden_dim= args.hidden_dim, num_classes= args.n_classes, heads= args.heads, pooling_ratio= args.pooling_ratio, walk_length= args.encoding_size, conv_type= args.convolution, attention= args.attention)


    for weight_root, _, weight_file in os.walk(args.checkpoint_weights):
        weights = os.path.join(weight_root, weight_file[0])
    checkpoint = torch.load(weights)
    graph_net.load_state_dict(checkpoint, strict=True)

    if use_gpu:
        graph_net.cuda()

    # load pickled graphs
    if args.encoding_size == 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)

    if args.encoding_size > 0:
        with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
            graph_dict = pickle.load(file)

    # load train/test split
    with open(current_directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as file:
        splits = pickle.load(file)

    test_ids = splits[args.test_fold]['Test']

    results = []

    for patient_id in args.test_ids:

        slide_embedding = graph_dict[patient_id]
        test_graph_loader = DataLoader(slide_embedding, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        if per_layer:

            attention_scores, actual_label, predicted_label = slide_att_scores_per_layer(graph_net, test_graph_loader, patient_id, loss_fn, n_classes=args.n_classes)
            #results.append([patient_id, actual_label.item(), predicted_label.item()])

            #with open(current_directory + f"/attn_score_dict_{args.graph_mode}_{args.test_fold}_{patient_id}_{args.dataset_name}.pkl", "wb") as file:
            #    pickle.dump(attention_scores, file)

            normalized_attention_scores = normalize_attention_scores_per_layer(attention_scores)
            for i in range(4):
                 create_heatmaps_per_layer(patient_id, i, normalized_attention_scores[patient_id][i], img_folders, args.heatmap_path, heatmap_list, args.path_to_patches, args.patch_size)

        else:

            attention_scores, actual_label, predicted_label = slide_att_scores(graph_net, test_graph_loader, patient_id, loss_fn, n_classes=args.n_classes)
            create_heatmaps(patient_id, attention_scores, img_folders, args.heatmap_path, heatmap_list, args.path_to_patches, args.patch_size)

            results.append([patient_id, actual_label.item(), predicted_label.item()])

            with open(current_directory + f"/attn_score_dict_{args.graph_mode}_{args.test_fold}_{patient_id}_{args.dataset_name}.pkl", "wb") as file:
                pickle.dump(attention_scores, file)

    df_results = pd.DataFrame(results, columns=['Patient_ID', 'Label', 'Predicted_label'])
    df_results.to_csv(current_directory + f"/predicted_results_{args.graph_mode}_{args.test_fold}_{args.dataset_name}.csv", index=False)


# %%

if __name__ == "__main__":

    graph_modes = ['krag']
    for graph in graph_modes:
        args = arg_parse()
        args.directory = "/data/scratch/wpw030/CAMELYON16/feature_dictionaries/"
        args.path_to_patches = "/data/scratch/wpw030/CAMELYON16/results_10/patches/"
        args.checkpoint_weights = "/data/scratch/wpw030/CAMELYON16/krag_test_checkpoints/"
        args.heatmap_path = "/data/scratch/wpw030/CAMELYON16/heatmaps_per_layer/"
        args.dataset_name = "CAMELYON16"
        args.embedding_net = 'resnet18'
        args.convolution = 'GAT'
        args.graph_mode = graph
        args.attention = False
        args.encoding_size = 20
        args.stain_type = 'H&E'
        args.test_fold = "Fold 0"
        args.test_ids = ['test_016', 'test_071', 'test_102']
        main(args)