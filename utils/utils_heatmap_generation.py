# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:12:44 2024

@author: AmayaGS
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import logging

import torch

# PyG
from torch_geometric.loader import DataLoader

# KRAG functions
from models.krag_heatmap_models import KRAG_Classifier
from train_test_loops.training_loop_heatmap import slide_att_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def extract_numbers(string):
    pattern = r'=(\d+)'
    numbers = re.findall(pattern, string)
    return [int(num) for num in numbers]


def already_processed(filename, heatmap_path, heatmap_type, test_fold):
    raw_file = f"{heatmap_path}/{filename}_raw_{heatmap_type}_fold_{test_fold}.png"
    smoothed_file = f"{heatmap_path}/{filename}_smoothed_{heatmap_type}_fold_{test_fold}.png"
    return os.path.exists(raw_file) and os.path.exists(smoothed_file)


def create_spatial_info_dict(df_filename):
    return {row['Patch_name']: ({'row1': row['Patch_coordinates'][2],
                                 'row2': row['Patch_coordinates'][3],
                                 'col1': row['Patch_coordinates'][0],
                                 'col2': row['Patch_coordinates'][1]},
                                row['norm_score'])
            for _, row in df_filename.iterrows()}


def create_canvas_and_att_img(spatial_info_dict, path_to_folder, patch_size):
    max_x = max(info[0]['row2'] for info in spatial_info_dict.values())
    max_y = max(info[0]['col2'] for info in spatial_info_dict.values())

    canvas = np.zeros((max_y + patch_size, max_x + patch_size, 3), dtype=np.uint8)
    att_img = np.zeros((max_y + patch_size, max_x + patch_size), dtype=np.float64)

    for patch_name, (coordinates, score) in spatial_info_dict.items():
        x1, x2, y1, y2 = coordinates['row1'], coordinates['row2'], coordinates['col1'], coordinates['col2']

        att_img[y1:y2, x1:x2] = score

        patch_image_path = os.path.join(path_to_folder, patch_name)

        if not os.path.exists(patch_image_path):
            logging.warning(f"File not found: {patch_image_path}")
            continue

        try:
            patch_image = np.array(Image.open(patch_image_path))
            if patch_image.shape[2] == 4:
                patch_image = patch_image[:, :, :3]
            canvas[y1:y2, x1:x2, :] = patch_image
        except Exception as e:
            logging.error(f"Error processing file {patch_image_path}: {str(e)}")

    return canvas, att_img


def plot_heatmap(canvas, att_img, smoothed_att, filename, heatmap_path, heatmap_type, test_fold):
    reconstructed_image_pil = Image.fromarray(canvas)

    # Raw heatmap
    raw_fig = plt.figure(figsize=(60, 20))
    plt.subplot(121)
    plt.title('Original image', size=60)
    plt.imshow(reconstructed_image_pil)

    plt.subplot(122)
    plt.title('Predicted heatmap', size=60)
    heatmap_ = plt.imshow(att_img, cmap=plt.cm.jet)
    plt.colorbar(heatmap_)
    plt.imshow(reconstructed_image_pil, alpha=0.4)

    plt.suptitle(f'{filename} {heatmap_type}', size=60)
    plt.savefig(f"{heatmap_path}/{filename}_raw_{heatmap_type}_fold_{test_fold}.png", dpi=raw_fig.dpi)
    plt.close()

    # Smoothed heatmap
    smoothed_fig = plt.figure(figsize=(60, 20))
    plt.subplot(121)
    plt.title('Original image', size=60)
    plt.imshow(reconstructed_image_pil)

    plt.subplot(122)
    plt.title('Smoothed heatmap', size=60)
    heatmap_ = plt.imshow(smoothed_att, cmap=plt.cm.jet)
    plt.imshow(reconstructed_image_pil, alpha=0.4)

    plt.suptitle(f'{filename} {heatmap_type}', size=60)
    plt.savefig(f"{heatmap_path}/{filename}_smoothed_{heatmap_type}_fold_{test_fold}.png",
                dpi=smoothed_fig.dpi)
    plt.close()


def process_scores(patient_id, scores, heatmap_path, path_to_patches, patch_size, test_fold, heatmap_type):
    grouped_scores = {}
    for patch_name, score in scores:
        filename = patch_name.split('_row1')[0]  # Extract filename from patch name
        if patient_id in filename:  # Only include filenames that contain the patient_id
            if filename not in grouped_scores:
                grouped_scores[filename] = []
            grouped_scores[filename].append((patch_name, score))

    for filename, file_scores in grouped_scores.items():
        if already_processed(filename, heatmap_path, heatmap_type, test_fold):
            logging.info(f"Skipping {filename} for {heatmap_type}, fold {test_fold} as it's already processed.")
            continue

        logging.info(f"Processing heatmap for file: {filename} for {heatmap_type}, fold {test_fold}")

        path_to_folder = os.path.join(path_to_patches, filename)

        if not os.path.exists(path_to_folder):
            logging.warning(f"Folder not found: {path_to_folder}")
            continue

        df_filename = pd.DataFrame(file_scores, columns=['Patch_name', 'norm_score'])
        df_filename['Filename'] = filename
        df_filename['Patch_coordinates'] = df_filename['Patch_name'].apply(extract_numbers)

        spatial_info_dict = create_spatial_info_dict(df_filename)
        canvas, att_img = create_canvas_and_att_img(spatial_info_dict, path_to_folder, patch_size)
        smoothed_att = gaussian_filter(att_img, sigma=200)

        plot_heatmap(canvas, att_img, smoothed_att, filename, heatmap_path, heatmap_type, test_fold)


def create_heatmaps(patient_id, attention_scores, heatmap_path, path_to_patches, patch_size, test_fold):
    os.makedirs(heatmap_path, exist_ok=True)

    # Process cumulative scores
    process_scores(patient_id, attention_scores['cumulative'], heatmap_path, path_to_patches, patch_size, test_fold,
                   "cumulative")

    # Process per-layer scores
    for layer, layer_scores in enumerate(attention_scores['per_layer']):
        process_scores(patient_id, layer_scores, heatmap_path, path_to_patches, patch_size, test_fold, f"layer_{layer}")


def process_fold(args, fold, test_ids, graph_dict, heatmap_path, loss_fn, current_directory):
    # Initialize the model
    graph_net = KRAG_Classifier(
        args.embedding_vector_size,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_classes,
        heads=args.heads,
        pooling_ratio=args.pooling_ratio,
        walk_length=args.encoding_size,
        conv_type=args.convolution,
        attention=args.attention
    )

    # Load model weights
    checkpoint_path = os.path.join(args.checkpoint_weights, f"best_val_models/checkpoint_fold_{fold}_bm.pth")
    checkpoint = torch.load(checkpoint_path)
    graph_net.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        graph_net.cuda()

    graph_net.eval()

    results = []

    for patient_id in test_ids:
        if patient_id not in graph_dict:
            logging.info(f"Warning: Patient ID {patient_id} not found in graph dictionary. Skipping.")
            continue

        slide_embedding = graph_dict[patient_id]
        test_graph_loader = DataLoader([slide_embedding], batch_size=1, shuffle=False, num_workers=args.num_workers)

        logging.info(f"Processing heatmaps for patient: {patient_id}")
        attention_scores, actual_label, predicted_label = slide_att_scores(graph_net,
                                                                           test_graph_loader,
                                                                           patient_id,
                                                                           loss_fn,
                                                                           n_classes=args.n_classes)

        results.append([patient_id, actual_label.item(), predicted_label.item()])
        # Save attention attention_scores
        with open(
                current_directory + f"/attn_score_dict_{args.graph_mode}_fold_{fold}_{patient_id}_{args.dataset_name}.pkl",
                "wb") as file:
            pickle.dump(attention_scores, file)

        # Create heatmaps
        create_heatmaps(patient_id,
                        attention_scores[patient_id],
                        heatmap_path,
                        args.path_to_patches,
                        args.patch_size,
                        fold)

    # Save results
    df_results = pd.DataFrame(results, columns=['Patient_ID', 'Label', 'Predicted_label'])
    df_results.to_csv(current_directory + f"/predicted_results_{args.graph_mode}_{fold}_{args.dataset_name}.csv",
                      index=False)

    logging.info(f"Heatmap generation completed for fold {fold}. Results saved.")