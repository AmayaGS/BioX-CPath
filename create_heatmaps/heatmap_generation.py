# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:12:44 2024

@author: AmayaGS
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy.ndimage import gaussian_filter

#%%

path_to_patches = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches"
image_filenames = os.listdir(path_to_patches)
patch_size = 224

with open(r"C:\Users\Amaya\Documents\PhD\MUSTANGv2\results\attn_score_dict_fold_9_RA.pkl", "rb") as file:
    attn_score = pickle.load(file)

test_ids = []
for filename in image_filenames:
    patient_id = filename.split("_")[0]
    test_ids.append(patient_id)
test_ids = list(np.unique(test_ids))

# %%

def extract_numbers(string):
    pattern = r'=(\d+)'
    numbers = re.findall(pattern, string)
    return [int(num) for num in numbers]

# %%
#attn_score["LOUV-R4RA-L974"]

df = []
for test_id in test_ids:
    if test_id in attn_score:
        attn_score_pid = attn_score[test_id]
        for idx in range(len(attn_score_pid[0])):
            if test_id in attn_score_pid[0][idx][0][0]:
                df.append([test_id, attn_score_pid[0][idx][0][0].split("._")[0], extract_numbers(attn_score_pid[0][idx][0][0].split("._")[1][:-4]), attn_score_pid[0][idx][0][0].replace("._x=", "_x="), attn_score_pid[0][idx][1]])
df = pd.DataFrame(df, columns=["Patient_ID", "Filename", "Patch_coordinates", "Patch_name", "Score"])

# %%

attn_heatmap_dict = {}

for filename in image_filenames:

    df_filename = df[df['Filename'] == filename]

    if df_filename.size != 0:

        # Normalize att score [0--1]
        df_filename.loc[:, 'norm_score'] = ((df_filename['Score'] - df_filename['Score'].min()) / (df_filename['Score'].max() - df_filename['Score'].min())).fillna(0)

        spatial_info_dict = {}

        for index, row in enumerate(df_filename.iterrows()):
            coordinates = df_filename['Patch_coordinates'].iloc[index]

            patch_name = df_filename['Patch_name'].iloc[index]

            spatial_info_dict[patch_name] = ({
                'x1': coordinates[2],
                'x2': coordinates[3],
                'y1': coordinates[0],
                'y2': coordinates[1]},
                df_filename['norm_score'].iloc[index])

        max_x = int(max(info[0]['x2'] for info in spatial_info_dict.values()))
        max_y = int(max(info[0]['y2'] for info in spatial_info_dict.values()))

        canvas = np.zeros((max_y + patch_size, max_x + patch_size, 3), dtype=np.uint8)
        att_img = np.zeros((max_y + patch_size, max_x + patch_size), dtype=np.float64)

        for patient_id, (coordinates, score) in spatial_info_dict.items():

            x1, x2, y1, y2 = coordinates['x1'], coordinates['x2'], coordinates['y1'], coordinates['y2']

            # att heatmap
            patch_score = np.ones((patch_size, patch_size))
            weighted_patch = patch_score * score
            att_img[y1:y2, x1:x2] = weighted_patch

            # reconstruct original image
            patch_image_path = os.path.join(path_to_patches, filename, patient_id)
            patch_image = np.array(Image.open(patch_image_path))
            if patch_image.shape[2] == 4:
                patch_image = patch_image[:, :, :3]

            canvas[y1:y2, x1:x2, :] = patch_image

        smoothed_att = gaussian_filter(att_img, sigma=100)
        attn_heatmap_dict[filename] = [canvas, att_img, smoothed_att]

    else:

        # Continue to the next iteration
        print("DataFrame is empty. Skipping to the next DataFrame.")
        continue

# %%

for k, v in attn_heatmap_dict.items():

    reconstructed_image_pil = Image.fromarray(v[0])
    original_heatmap = v[1]
    smoothed_heatmap = v[2]

    ### METHOD 1: heatmap without smoothing ###

    plt.figure(figsize=(60, 20))

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

    # ### METHOD 2: heatmap with smoothing ###

    plt.figure(figsize=(60, 20))
    plt.subplot(121)
    plt.title('Original image', size=60)
    plt.imshow(reconstructed_image_pil)

    plt.subplot(122)
    plt.title('Smoothed heatmap', size=60)
    heatmap_ = plt.imshow(smoothed_heatmap, cmap=plt.cm.jet)
    #plt.colorbar(heatmap_, orientation="horizontal")
    plt.imshow(reconstructed_image_pil, alpha=0.4)

    plt.suptitle(f'{k}', size=60)
    plt.show()

# %%