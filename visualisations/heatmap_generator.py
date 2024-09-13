# heatmap_generator.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.KRAG_heatmap_model import KRAG_Classifier
from train_test_loops.krag_heatmap_loop import heatmap_scores


class HeatmapGenerator:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_heatmaps(self, patient_id, patient_dir, attention_scores, fold):
        self._process_scores(patient_id, patient_dir, attention_scores[patient_id]['cumulative'], "cumulative", fold)
        for layer, layer_scores in enumerate(attention_scores[patient_id]['per_layer']):
            self._process_scores(patient_id, patient_dir, layer_scores, f"layer_{layer}", fold)

    def _process_scores(self, patient_id, patient_dir, scores, heatmap_type, fold):
        grouped_scores = self._group_scores(patient_id, scores)
        for filename, file_scores in grouped_scores.items():
            if self._already_processed(filename, patient_dir, heatmap_type, fold):
                self.logger.info(f"Skipping {filename} for {heatmap_type}, fold {fold} as it's already processed.")
                continue

            self.logger.info(f"Processing heatmap for file: {filename} for {heatmap_type}, fold {fold}")

            df_filename = pd.DataFrame(file_scores, columns=['Patch_name', 'norm_score'])
            df_filename['Filename'] = filename
            df_filename['Patch_coordinates'] = df_filename['Patch_name'].apply(self._extract_numbers)

            spatial_info_dict = self._create_spatial_info_dict(df_filename)
            canvas, att_img = self._create_canvas_and_att_img(spatial_info_dict, filename)
            smoothed_att = gaussian_filter(att_img, sigma=100)

            self._plot_heatmap(patient_dir, canvas, att_img, smoothed_att, filename, heatmap_type, fold)

    def _group_scores(self, patient_id, scores):
        grouped_scores = {}
        for patch_name, score in scores:
            filename = patch_name.split('_row1')[0]
            if patient_id in filename:
                if filename not in grouped_scores:
                    grouped_scores[filename] = []
                grouped_scores[filename].append((patch_name, score))
        return grouped_scores

    def _create_spatial_info_dict(self, df_filename):
        return {row['Patch_name']: ({'row1': row['Patch_coordinates'][2],
                                     'row2': row['Patch_coordinates'][3],
                                     'col1': row['Patch_coordinates'][0],
                                     'col2': row['Patch_coordinates'][1]},
                                    row['norm_score'])
                for _, row in df_filename.iterrows()}

    def _create_canvas_and_att_img(self, spatial_info_dict, filename):
        max_x = max(info[0]['row2'] for info in spatial_info_dict.values())
        max_y = max(info[0]['col2'] for info in spatial_info_dict.values())

        canvas = np.zeros((max_y + self.args.patch_size, max_x + self.args.patch_size, 3), dtype=np.uint8)
        att_img = np.zeros((max_y + self.args.patch_size, max_x + self.args.patch_size), dtype=np.float64)

        for patch_name, (coordinates, score) in spatial_info_dict.items():
            x1, x2, y1, y2 = coordinates['row1'], coordinates['row2'], coordinates['col1'], coordinates['col2']

            att_img[y1:y2, x1:x2] = score

            patch_image_path = os.path.join(self.args.path_to_patches, filename, patch_name)

            if not os.path.exists(patch_image_path):
                self.logger.warning(f"File not found: {patch_image_path}")
                continue

            try:
                patch_image = np.array(Image.open(patch_image_path))
                if patch_image.shape[2] == 4:
                    patch_image = patch_image[:, :, :3]
                canvas[y1:y2, x1:x2, :] = patch_image
            except Exception as e:
                self.logger.error(f"Error processing file {patch_image_path}: {str(e)}")

        return canvas, att_img

    def _plot_heatmap(self, patient_dir, canvas, att_img, smoothed_att, filename, heatmap_type, fold):
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
        plt.savefig(f"{patient_dir}/{filename}_raw_{heatmap_type}_fold_{fold}.png", dpi=raw_fig.dpi)
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
        plt.savefig(f"{patient_dir}/{filename}_smoothed_{heatmap_type}_fold_{fold}.png",
                    dpi=smoothed_fig.dpi)
        plt.close()

    def _already_processed(self, patient_dir, filename, heatmap_type, fold):
        raw_file = f"{patient_dir}/{filename}_raw_{heatmap_type}_fold_{fold}.png"
        smoothed_file = f"{patient_dir}/{filename}_smoothed_{heatmap_type}_fold_{fold}.png"
        return os.path.exists(raw_file) and os.path.exists(smoothed_file)

    def _extract_numbers(self, string):
        import re
        pattern = r'=(\d+)'
        numbers = re.findall(pattern, string)
        return [int(num) for num in numbers]