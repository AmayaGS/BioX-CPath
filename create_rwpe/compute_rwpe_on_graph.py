# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:29:52 2024

@author: AmayaGS
"""

# misc
import pickle

# pytorch
import torch

# torch geometric
import torch_geometric.transforms as T

# MUSTANG functions
from utils.auxiliary_functions import seed_everything

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc
gc.enable()

def add_pe_to_graph(loader, walk_length):

    loader_PE = {}

    for batch_idx, (patient_ID, graph_object) in enumerate(loader.items()):

        data, label, folder_ids, filenames = graph_object

        transform = T.AddRandomWalkPE(walk_length)
        data = transform(data)

        data.x = torch.cat([data.x, data.random_walk_pe], dim=1)

        loader_PE[patient_ID] = [data, label, folder_ids, filenames]

        del data, label, folder_ids, filenames, patient_ID, graph_object
        gc.collect()

    return loader_PE



def compute_rwpe(args):

    seed_everything(args.seed)

    current_directory = args.directory

    # load pickled graphs
    with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
        graph_dict = pickle.load(file)

    # adding RWPE here
    graph_dict = add_pe_to_graph(graph_dict, args.encoding_size)

    with open(current_directory + f"/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(graph_dict, file)  # encode dict into Pickle

#
# if __name__ == "__main__":
#     args = arg_parse()
#     graph_types = ['krag']
#     #stains = ['HE', 'CD20', 'CD138', 'CD3', 'CD21']
#     #stains = ['H&E','CD68', 'CD20', 'CD138']
#     stains = ['H&E']
#     #stains = ['all']
#     for graph_type in graph_types:
#         for stain in stains:
#             args.dataset_name = "CAMELYON16"
#             args.directory = "/data/scratch/wpw030/CAMELYON16/results_5/"
#             args.embedding_net = 'resnet18'
#             args.graph_mode = graph_type
#             args.encoding_size = 20
#             args.stains = stain
#             main(args)
#
# # # %%