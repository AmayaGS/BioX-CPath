# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:12:44 2024

@author: AmayaGS
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy.ndimage import gaussian_filter
import gc



def extract_numbers(string):
    pattern = r'=(\d+)'
    numbers = re.findall(pattern, string)
    return [int(num) for num in numbers]

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

def create_heatmaps_per_layer(patient_id, i, attn_score, img_folders, heatmap_path, heatmap_list, path_to_patches, patch_size):

    os.makedirs(heatmap_path + f"_{i}", exist_ok=True)

    for filename in img_folders:
        if filename not in heatmap_list:
            if patient_id in filename:

                attn_heatmap_dict = {}

                path_to_folder = os.path.join(path_to_patches, filename)

                df = []
                #attn_score_pid = attn_score[patient_id]
                attn_score_pid = attn_score
                for idx in range(len(attn_score_pid)):
                    df.append([patient_id, attn_score_pid[idx][0][0].split('_row1')[0], extract_numbers(attn_score_pid[idx][0][0]), attn_score_pid[idx][0][0], attn_score_pid[idx][1]])
                df = pd.DataFrame(df, columns=["Patient_ID", "Filename", "Patch_coordinates", "Patch_name", "norm_score"])

                df_filename = df[df['Filename'] == filename]
                df_filename.loc[:, 'norm_score'] = df_filename.loc[:, 'norm_score'].fillna(0)

                spatial_info_dict = {}
                print(filename)

                for index, row in enumerate(df_filename.iterrows()):
                    coordinates = df_filename['Patch_coordinates'].iloc[index]

                    patch_name = df_filename['Patch_name'].iloc[index]

                    spatial_info_dict[patch_name] = ({
                        'row1': coordinates[2],
                        'row2': coordinates[3],
                        'col1': coordinates[0],
                        'col2': coordinates[1]},
                        df_filename['norm_score'].iloc[index])

                max_x = int(max(info[0]['row2'] for info in spatial_info_dict.values()))
                max_y = int(max(info[0]['col2'] for info in spatial_info_dict.values()))

                canvas = np.zeros((max_y + patch_size, max_x + patch_size, 3), dtype=np.uint8)
                att_img = np.zeros((max_y + patch_size, max_x + patch_size), dtype=np.float64)

                for patch_name, (coordinates, score) in spatial_info_dict.items():

                    x1, x2, y1, y2 = coordinates['row1'], coordinates['row2'], coordinates['col1'], coordinates['col2']

                    # att heatmap
                    patch_score = np.ones((patch_size, patch_size))
                    weighted_patch = patch_score * score
                    att_img[y1:y2, x1:x2] = weighted_patch

                    # reconstruct original image
                    patch_image_path = os.path.join(path_to_folder, patch_name)
                    patch_image = np.array(Image.open(patch_image_path))
                    if patch_image.shape[2] == 4:
                        patch_image = patch_image[:, :, :3]

                    canvas[y1:y2, x1:x2, :] = patch_image

                smoothed_att = gaussian_filter(att_img, sigma=200)
                attn_heatmap_dict[filename] = [canvas, att_img, smoothed_att]

                for k, v in attn_heatmap_dict.items():

                    reconstructed_image_pil = Image.fromarray(v[0])
                    original_heatmap = v[1]
                    smoothed_heatmap = v[2]

                    ### METHOD 1: heatmap without smoothing ###
                    raw_fig = plt.figure(figsize=(60, 20))
                    plt.subplot(121)
                    plt.title('Original image', size=60)
                    plt.imshow(reconstructed_image_pil)

                    plt.subplot(122)
                    plt.title('Predicted heatmap', size=60)
                    heatmap_ = plt.imshow(original_heatmap, cmap=plt.cm.jet)
                    plt.colorbar(heatmap_)
                    plt.imshow(reconstructed_image_pil, alpha=0.4)

                    plt.suptitle(f'{k}', size=60)
                    plt.show()
                    raw_fig.savefig(heatmap_path + f"_{i}" + f"/{k}_raw_{i}.png", dpi=raw_fig.dpi)
                    plt.close()
                    gc.collect()

                    # ### METHOD 2: heatmap with smoothing ###
                    smoothed_fig = plt.figure(figsize=(60, 20))
                    plt.subplot(121)
                    plt.title('Original image', size=60)
                    plt.imshow(reconstructed_image_pil)

                    plt.subplot(122)
                    plt.title('Smoothed heatmap', size=60)
                    heatmap_ = plt.imshow(smoothed_heatmap, cmap=plt.cm.jet)
                    plt.imshow(reconstructed_image_pil, alpha=0.4)

                    plt.suptitle(f'{k}', size=60)
                    plt.show()
                    smoothed_fig.savefig(heatmap_path + f"_{i}" + f"/{k}_smoothed_{i}.png", dpi=smoothed_fig.dpi)
                    plt.close()
                    gc.collect()