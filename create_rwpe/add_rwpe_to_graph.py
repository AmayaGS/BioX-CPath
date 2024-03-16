# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:29:52 2024

@author: AmayaGS
"""

# misc
import os
import os.path
from pathlib import Path
import pickle
import argparse

# pytorch
import torch

# torch geometric
import torch_geometric.transforms as T

# KRAG functions
from auxiliary_functions import seed_everything

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc
gc.enable()


def arg_parse():

    parser = argparse.ArgumentParser(description="self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", help="Dataset name")
    parser.add_argument("--data_directory", type=str, default="/data/scratch/wpw030/KRAG", help="Location of patient label df and extracted patches df. Embeddings and graphs dictionaries will be kept here.")
    parser.add_argument("--embedding_net", type=str, default="vgg16", choices=['resnet18', 'vgg16', 'convnext'], help="feature extraction network used")
    parser.add_argument("--graph_mode", type=str, default="krag", choices=['krag', 'rag', 'knn'], help="type of graph used")
    parser.add_argument("--positional_encoding", default=True, help="Add Random Walk positional encoding to the graph")
    parser.add_argument("--encoding_size", type=float, default=20, help="Size Random Walk positional encoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def add_pe_to_graph(loader, walk_length):

    loader_PE = {}

    for batch_idx, (patient_ID, graph_object) in enumerate(loader.items()):

        data, label, folder_ids, filenames = graph_object

        transform = T.AddRandomWalkPE(walk_length)
        data = transform(data)

        data.x = torch.cat([data.x, data.random_walk_pe], dim=1)

        loader_PE[patient_ID] = [data.to('cpu'), label.to('cpu'), folder_ids, filenames]

        del data, label, folder_ids, filenames, patient_ID, graph_object
        gc.collect()

    return loader_PE


def main(args):

    seed_everything(args.seed)

    data_directory = args.data_directory

    # load pickled graphs
    with open(data_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}.pkl", "rb") as file:
        graph_dict = pickle.load(file)

    # adding RWPE here
    graph_dict = add_pe_to_graph(graph_dict, args.encoding_size)

    with open(data_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}.pkl", "wb") as file:
        pickle.dump(graph_dict, file)  # encode dict into Pickle




if __name__ == "__main__":
    args = arg_parse()
    args.data_directory = r"C:/Users/Amaya/Documents/PhD/MUSTANGv2/min_code_krag/data"
    args.dataset_name = "RA"
    args.graph_mode = 'krag'
    args.positional_encoding = True
    args.encoding_size = 5
    main(args)