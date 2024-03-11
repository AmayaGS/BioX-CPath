# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:14:46 2023

@author: AmayaGS

"""

# Misc
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
import random
import pickle

# sklearn
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedShuffleSplit

# PyTorch
import torch
from torch_geometric.data import Data

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# Define collate function
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



def create_stratified_splits(extracted_patches, patient_labels, patient_id, label, train_fraction, seed, dataset_name):

    # merging patches with the patient labels
    df = pd.merge(extracted_patches, patient_labels, on=patient_id)

    # drop duplicates to obtain the actual patient IDs that have a label assigned by the pathologist
    df_labels = df.drop_duplicates(subset="Patient_ID")

    # stratified split on labels
    sss = StratifiedShuffleSplit(n_splits=10, test_size=1 - train_fraction, random_state=seed)
    sss.get_n_splits(df_labels[patient_id], df_labels[label])

    # creating a dictionary which keeps a list of the Patient IDs from the stratified training splits. Outer key is Fold, inner key is Train/Test.
    fold_dictionary = {}

    for i, (train_index, test_index) in enumerate(sss.split(df_labels[patient_id], df_labels[label])):
        fold_name = f"Fold {i}"
        fold_dictionary[fold_name] = {
            "Train": list(df_labels.iloc[train_index][patient_id]),
            "Test": list(df_labels.iloc[test_index][patient_id])
        }

    with open(f"train_test_strat_splits_{dataset_name}.pkl", "wb") as file:
        pickle.dump(fold_dictionary, file)  # encode dict into Pickle



def string_to_int_list(s):
    # Remove brackets and split by spaces
    numbers = s.strip('[]').split()
    # Convert each substring to integer
    return [int(num) for num in numbers]


# Function to create spatial adjacency matrix
def create_adjacency_matrix(patches):

    num_patches = len(patches)
    adjacency_matrix = np.zeros((num_patches, num_patches), dtype=int)

    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            patch1 = patches[i]
            patch2 = patches[j]

            # Check if patches are adjacent horizontally, vertically, or diagonally
            if (patch1[0] <= patch2[1] and patch1[1] >= patch2[0] and
                patch1[2] <= patch2[3] and patch1[3] >= patch2[2]):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix



def create_embedding_graphs(embedding_net, loader, k=7, include_self=True):


    embedding_dict = dict()
    knn = dict()
    rag = dict()
    krag = dict()

    embedding_net.eval()
    with torch.no_grad():

        for patient_ID, slide_loader in loader.items():

            patient_embedding = []
            patient_ids = []
            folder_ids  = []
            filenames = []
            coordinates = []

            for patch in slide_loader:

                inputs, label, patient_id, folder_id, file_name, coordinate = patch

                label = label[0].unsqueeze(0)
                patient_ID = patient_id[0]
                folder_ID = folder_id[0]
                coordinate = coordinate[0]

                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label

                embedding = embedding_net(inputs)
                embedding = embedding.to('cpu')
                embedding = embedding.squeeze(0).squeeze(0)

                patient_embedding.append(embedding)
                patient_ids.append(patient_ID)
                folder_ids.append(folder_ID)
                filenames.append(file_name)
                coordinates.append(coordinate)

            patient_embedding = torch.stack(patient_embedding)

            # Embedding dictionary
            embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu'), list(np.unique(folder_ids)), filenames]

            #if graph_mode == 'rag':
            # Region-adjacency dictionary here
            # this spatial adjacency is on the patient_ID, not the individual image level - hence there can be more than 8 edges. This design choice could be reviewed. Most images from a same patient correspond to slices and therefore align spatially.
            coord = [string_to_int_list(s) for s in coordinates]
            spatial_adjacency_matrix = create_adjacency_matrix(coord)
            spatial_edge_index = (torch.tensor(spatial_adjacency_matrix) > 0).nonzero().t()
            spatial_data = Data(x=patient_embedding, edge_index=spatial_edge_index)
            rag[patient_ID] = [spatial_data.to('cpu'), label.to('cpu'), list(np.unique(folder_ids)), filenames]

            #if graph_mode == 'knn':
            # KNN dictionary here
            knn_graph = kneighbors_graph(patient_embedding, k, include_self=include_self)
            knn_edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            knn_data = Data(x=patient_embedding, edge_index=knn_edge_index)
            knn[patient_ID] = [knn_data.to('cpu'), label.to('cpu'), list(np.unique(folder_ids)), filenames]

            #if graph_mode == 'krag':
            # KNN + KRAG graph dictionary here
            knn_graph = kneighbors_graph(patient_embedding, k, include_self=include_self)
            coord = [string_to_int_list(s) for s in coordinates]
            spatial_adjacency_matrix = create_adjacency_matrix(coord)
            knn_spatial_adj = knn_graph.A  +  spatial_adjacency_matrix
            knn_spatial_adj[np.where(knn_spatial_adj > 1)] = 1
            knn_spatial_edge_index = (torch.tensor(knn_spatial_adj) > 0).nonzero().t()
            knn_spatial_data = Data(x=patient_embedding, edge_index=knn_spatial_edge_index)
            krag[patient_ID] = [knn_spatial_data.to('cpu'), label.to('cpu'), list(np.unique(folder_ids)), filenames]


    return embedding_dict, rag, knn, krag