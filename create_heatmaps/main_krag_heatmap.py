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
from training_loops.training_loop_heatmap import slide_att_scores, slide_att_scores_per_layer
from utils.auxiliary_functions import seed_everything
from models.krag_heatmap_models import KRAG_Classifier, KRAG_Classifier_per_layer
from utils.utils_heatmap_generation import create_heatmaps, create_heatmaps_per_layer, normalize_attention_scores_per_layer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()


def heatmap_generation(args):

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
