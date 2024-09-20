# Misc
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import ast
import json
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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#Define collate function
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Function to create spatial adjacency matrix
def create_adjacency_matrix(patches):

    num_patches = len(patches)
    adjacency_matrix = np.zeros((num_patches, num_patches), dtype=int)

    for i in range(num_patches):
        # Add self-loop
        adjacency_matrix[i, i] = 1

        for j in range(i + 1, num_patches):
            patch1 = patches[i]
            patch2 = patches[j]

            # Check if patches are adjacent horizontally, vertically, or diagonally
            if (patch1[0] <= patch2[1] and patch1[1] >= patch2[0] and
                patch1[2] <= patch2[3] and patch1[3] >= patch2[2]):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix


def create_krag_graph(rag_edge_index, knn_edge_index, edge_types):
    # Convert edge indices to sets for efficient lookup
    rag_edges = set(map(tuple, rag_edge_index.t().tolist()))
    knn_edges = set(map(tuple, knn_edge_index.t().tolist()))

    # Find union of all edges
    all_edges = sorted(rag_edges.union(knn_edges))

    # Create new edge index and attribute tensors
    krag_edge_index = torch.tensor(list(all_edges), dtype=torch.long).t()
    krag_edge_attr = torch.zeros((len(all_edges), 1), dtype=torch.long)

    # Determine edge types in one pass
    for i, edge in enumerate(all_edges):
        in_rag = edge in rag_edges
        in_knn = edge in knn_edges
        if in_rag and in_knn:
            krag_edge_attr[i] = edge_types['BOTH']
        elif in_rag:
            krag_edge_attr[i] = edge_types['RAG']
        else:  # must be in knn
            krag_edge_attr[i] = edge_types['KNN']

    return krag_edge_index, krag_edge_attr


def collect_graph_statistics(embedding_dict, knn, rag, krag, stain_types, edge_types):
    statistics = {
        'patient_level': {},
        'dataset_level': {
            'num_patients': 0,
            'num_slides': 0,
            'total_patches': 0,
            'patches_per_stain': {stain: 0 for stain in stain_types.keys()},
            'total_knn_edges': 0,
            'total_rag_edges': 0,
            'total_both_edges': 0,
            'total_combined_edges': 0
        }
    }

    slides_per_patient = []
    stains_per_patient = []
    patches_per_patient = []
    knn_edges_per_patient = []
    rag_edges_per_patient = []
    both_edges_per_patient = []
    combined_edges_per_patient = []

    for patient_id, patient_data in embedding_dict.items():
        patches_per_slide = {}
        for filename in patient_data[2]['filenames']:
            slide_id = filename[0].split('_')[0]  # Adjust this split based on your actual filename format
            if slide_id not in patches_per_slide:
                patches_per_slide[slide_id] = 0
            patches_per_slide[slide_id] += 1

        slides_per_patient.append(len(patches_per_slide))
        unique_stains = set(patient_data[2]['stains'])
        stains_per_patient.append(len(unique_stains))

        # Count edges by type
        krag_data = krag[patient_id][0]
        edge_attr = krag_data.edge_attr.cpu().numpy().flatten()
        knn_count = np.sum(edge_attr == edge_types['KNN'])
        rag_count = np.sum(edge_attr == edge_types['RAG'])
        both_count = np.sum(edge_attr == edge_types['BOTH'])
        combined_count = knn_count + rag_count + both_count

        patient_stats = {
            'patient_id': patient_id,
            'num_slides': slides_per_patient[-1],
            'num_stains': stains_per_patient[-1],
            'total_patches': len(patient_data[0]),
            'patches_per_slide': patches_per_slide,
            'patches_per_stain': {stain: patient_data[2]['stains'].count(stain_types[stain]) for stain in
                                  stain_types.keys()},
            'num_knn_edges': knn_count,
            'num_rag_edges': rag_count,
            'num_both_edges': both_count,
            'num_combined_edges': combined_count
        }

        statistics['patient_level'][patient_id] = patient_stats

        # Update dataset-level statistics
        statistics['dataset_level']['num_patients'] += 1
        statistics['dataset_level']['num_slides'] += patient_stats['num_slides']
        statistics['dataset_level']['total_patches'] += patient_stats['total_patches']
        statistics['dataset_level']['total_knn_edges'] += knn_count
        statistics['dataset_level']['total_rag_edges'] += rag_count
        statistics['dataset_level']['total_both_edges'] += both_count
        statistics['dataset_level']['total_combined_edges'] += combined_count

        for stain, count in patient_stats['patches_per_stain'].items():
            statistics['dataset_level']['patches_per_stain'][stain] += count

        patches_per_patient.append(patient_stats['total_patches'])
        knn_edges_per_patient.append(knn_count)
        rag_edges_per_patient.append(rag_count)
        both_edges_per_patient.append(both_count)
        combined_edges_per_patient.append(combined_count)

    # Calculate dataset-level statistics
    statistics['dataset_level']['avg_slides_per_patient'] = np.mean(slides_per_patient)
    statistics['dataset_level']['std_slides_per_patient'] = np.std(slides_per_patient)
    statistics['dataset_level']['avg_stains_per_patient'] = np.mean(stains_per_patient)
    statistics['dataset_level']['std_stains_per_patient'] = np.std(stains_per_patient)
    statistics['dataset_level']['avg_patches_per_patient'] = np.mean(patches_per_patient)
    statistics['dataset_level']['std_patches_per_patient'] = np.std(patches_per_patient)
    statistics['dataset_level']['avg_knn_edges_per_patient'] = np.mean(knn_edges_per_patient)
    statistics['dataset_level']['std_knn_edges_per_patient'] = np.std(knn_edges_per_patient)
    statistics['dataset_level']['avg_rag_edges_per_patient'] = np.mean(rag_edges_per_patient)
    statistics['dataset_level']['std_rag_edges_per_patient'] = np.std(rag_edges_per_patient)
    statistics['dataset_level']['avg_both_edges_per_patient'] = np.mean(both_edges_per_patient)
    statistics['dataset_level']['std_both_edges_per_patient'] = np.std(both_edges_per_patient)
    statistics['dataset_level']['avg_combined_edges_per_patient'] = np.mean(combined_edges_per_patient)
    statistics['dataset_level']['std_combined_edges_per_patient'] = np.std(combined_edges_per_patient)

    for stain in stain_types.keys():
        statistics['dataset_level'][f'avg_patches_per_patient_{stain}'] = np.mean(
            [patient_stats['patches_per_stain'][stain] for patient_stats in statistics['patient_level'].values()])
        statistics['dataset_level'][f'std_patches_per_patient_{stain}'] = np.std(
            [patient_stats['patches_per_stain'][stain] for patient_stats in statistics['patient_level'].values()])

    return statistics


def save_graph_statistics(statistics, output_dir):
    # Save patient-level statistics
    patient_df = pd.DataFrame.from_dict(statistics['patient_level'], orient='index')
    patient_df.reset_index(inplace=True)
    patient_df.rename(columns={'index': 'patient_id'}, inplace=True)

    # Convert patches_per_slide dictionary to a JSON string
    patient_df['patches_per_slide'] = patient_df['patches_per_slide'].apply(json.dumps)

    # Round numeric columns to 2 decimal places
    numeric_columns = ['num_slides', 'num_stains', 'total_patches', 'num_knn_edges', 'num_rag_edges', 'num_both_edges',
                       'num_combined_edges']
    patient_df[numeric_columns] = patient_df[numeric_columns].round(2)

    patient_df.to_csv(os.path.join(output_dir, 'patient_level_statistics.csv'), index=False)

    # Save dataset-level statistics
    dataset_stats = statistics['dataset_level']
    dataset_df = pd.DataFrame({
        'Metric': [
                      'Number of Patients',
                      'Number of Slides',
                      'Slides per Patient',
                      'Stains per Patient',
                      'Total Patches',
                      'Patches per Patient',
                      'KNN Edges per Patient',
                      'RAG Edges per Patient',
                      'BOTH Edges per Patient',
                      'Combined Edges per Patient'
                  ] + [f'Patches per Patient ({stain})' for stain in dataset_stats['patches_per_stain'].keys()],
        'Value': [
                     dataset_stats['num_patients'],
                     dataset_stats['num_slides'],
                     dataset_stats['avg_slides_per_patient'],
                     dataset_stats['avg_stains_per_patient'],
                     dataset_stats['total_patches'],
                     dataset_stats['avg_patches_per_patient'],
                     dataset_stats['avg_knn_edges_per_patient'],
                     dataset_stats['avg_rag_edges_per_patient'],
                     dataset_stats['avg_both_edges_per_patient'],
                     dataset_stats['avg_combined_edges_per_patient']
                 ] + [dataset_stats[f'avg_patches_per_patient_{stain}'] for stain in
                      dataset_stats['patches_per_stain'].keys()],
        'Std': [
                   '-',
                   '-',
                   dataset_stats['std_slides_per_patient'],
                   dataset_stats['std_stains_per_patient'],
                   '-',
                   dataset_stats['std_patches_per_patient'],
                   dataset_stats['std_knn_edges_per_patient'],
                   dataset_stats['std_rag_edges_per_patient'],
                   dataset_stats['std_both_edges_per_patient'],
                   dataset_stats['std_combined_edges_per_patient']
               ] + [dataset_stats[f'std_patches_per_patient_{stain}'] for stain in
                    dataset_stats['patches_per_stain'].keys()],
        'Total': [
                     dataset_stats['num_patients'],
                     dataset_stats['num_slides'],
                     '-',
                     '-',
                     dataset_stats['total_patches'],
                     '-',
                     dataset_stats['total_knn_edges'],
                     dataset_stats['total_rag_edges'],
                     dataset_stats['total_both_edges'],
                     dataset_stats['total_combined_edges']
                 ] + [dataset_stats['patches_per_stain'][stain] for stain in dataset_stats['patches_per_stain'].keys()]
    })

    # Round numeric values to 2 decimal places
    numeric_columns = ['Value', 'Std', 'Total']
    dataset_df[numeric_columns] = dataset_df[numeric_columns].apply(pd.to_numeric, errors='ignore').round(2)

    dataset_df.to_csv(os.path.join(output_dir, 'dataset_level_statistics.csv'), index=False)

    print(f"Statistics saved to {output_dir}")


def create_embedding_graphs(embedding_net, loader, k, include_self, stain_types, edge_types):

    embedding_dict = dict()
    knn = dict()
    rag = dict()
    krag = dict()

    embedding_net.eval()
    with torch.no_grad():
        for patient_ID, slide_loader in loader.items():
            patient_embedding = []
            patient_ids = []
            folder_ids = []
            filenames = []
            coordinates = []
            stains = []

            for patch in slide_loader:
                inputs, label, patient_id, folder_id, file_name, coordinate, stain = patch
                stain = stain[0]

                stains.append(stain_types[stain])

                label = label[0].unsqueeze(0)
                patient_ID = patient_id[0]
                folder_ID = folder_id[0]
                coordinate = coordinate[0]

                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()

                embedding = embedding_net(inputs)
                embedding = embedding.to('cpu')
                embedding = embedding.squeeze(0).squeeze(0)

                patient_embedding.append(embedding)
                patient_ids.append(patient_ID)
                folder_ids.append(folder_ID)
                filenames.append(file_name)
                coordinates.append(coordinate)

            patient_embedding = torch.stack(patient_embedding)
            node_attr = torch.tensor(stains, dtype=torch.long)

            # Create a dictionary to store patch metadata
            patch_metadata = {
                'folder_ids': list(np.unique(folder_ids)),
                'filenames': filenames,
                'coordinates': coordinates,
                'stains': stains
            }

            # Embedding dictionary
            embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu'), patch_metadata]

            # Region-adjacency dictionary
            # this spatial adjacency is on the patient_ID, not the individual image level - hence there can be more than 8 edges.
            # This design choice could be reviewed. Most images from a same patient correspond to slices and therefore aligned spatially.
            coord = [ast.literal_eval(s) for s in coordinates]
            rag_adj = torch.tensor(create_adjacency_matrix(coord), dtype=torch.float)
            rag_edge_index = rag_adj.nonzero().t()
            rag_edge_attr = torch.full((rag_edge_index.size(1), 1), edge_types['RAG'], dtype=torch.long)
            rag_data = Data(x=patient_embedding, edge_index=rag_edge_index, edge_attr=rag_edge_attr, node_attr=node_attr)
            rag[patient_ID] = [rag_data.to('cpu'), label.to('cpu'), patch_metadata]

            # KNN dictionary
            num_samples = patient_embedding.shape[0]
            k_adjusted = min(k, num_samples - 1 if include_self else num_samples)
            knn_adj = torch.tensor(kneighbors_graph(patient_embedding, k_adjusted, include_self=include_self).A, dtype=torch.float)
            knn_edge_index = knn_adj.nonzero().t()
            knn_edge_attr = torch.full((knn_edge_index.size(1), 1), edge_types['KNN'], dtype=torch.long)
            knn_data = Data(x=patient_embedding, edge_index=knn_edge_index, edge_attr=knn_edge_attr, node_attr=node_attr)
            knn[patient_ID] = [knn_data.to('cpu'), label.to('cpu'), patch_metadata]

            # KRAG graph
            krag_edge_index, krag_edge_attr = create_krag_graph(rag_edge_index, knn_edge_index, edge_types)
            krag_data = Data(x=patient_embedding, edge_index=krag_edge_index, edge_attr=krag_edge_attr, node_attr=node_attr)
            krag[patient_ID] = [krag_data.to('cpu'), label.to('cpu'), patch_metadata]

    statistics = collect_graph_statistics(embedding_dict, knn, rag, krag, stain_types, edge_types)

    return embedding_dict, knn, rag, krag, statistics