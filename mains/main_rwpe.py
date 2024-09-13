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
from utils.setup_utils import seed_everything

use_gpu = torch.cuda.is_available()

import gc
gc.enable()

def add_pe_to_graph(loader, walk_length):

    loader_PE = {}

    for batch_idx, (patient_ID, graph_object) in enumerate(loader.items()):

        data, label, metadata = graph_object

        transform = T.AddRandomWalkPE(walk_length)
        data = transform(data)

        data.x = torch.cat([data.x, data.random_walk_pe], dim=1)

        loader_PE[patient_ID] = [data, label, metadata]

        del data, label, metadata, patient_ID, graph_object
        gc.collect()

    return loader_PE


def compute_rwpe(args, logger):

    seed_everything(args.seed)

    # load pickled graphs
    with open(args.directory + f"/dictionaries/{args.graph_mode}_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "rb") as file:
        graph_dict = pickle.load(file)

    # adding RWPE here
    graph_dict = add_pe_to_graph(graph_dict, args.encoding_size)

    with open(args.directory + f"/dictionaries/{args.graph_mode}_dict_{args.dataset_name}_positional_encoding_{args.encoding_size}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(graph_dict, file)  # encode dict into Pickle
