# heatmap_generator.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn


class HeatmapGenerator:
    def __init__(self, args, results_dir, logger):
        self.args = args
        self.results_dir = results_dir
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_heatmaps(self, all_patient_data, fold_dir, fold):
        for patient in all_patient_data.keys():

            patient_dir = os.path.join(fold_dir, patient)
            os.makedirs(patient_dir, exist_ok=True)
            patient_data = all_patient_data[patient]
            folder_ids = patient_data['metadata']['folder_ids']
            cumulative_scores = patient_data['attention_scores'][patient]['cumulative']
            layer_scores = patient_data['attention_scores'][patient]['per_layer']
            label = patient_data['actual_label']
            predicted_label = patient_data['predicted_label']
            str_label = self.args.label_dict[str(label)]

            for folder in folder_ids:
                self.logger.info(f"Processing heatmap for patient: {patient}, stain: {folder}")
                spatial_info_dict = self._create_spatial_info_dict(patient_data, cumulative_scores, folder)

                canvas, att_img = self._create_canvas_and_att_img(spatial_info_dict, folder)

                self.plot_patch_heatmap(patient, patient_dir, canvas, att_img, folder, str_label, fold,
                                        spatial_info_dict, self.args.patch_size, self.args.path_to_patches,
                                        heatmap_type="cumulative")

                for layer, layer_score in enumerate(layer_scores):
                    self.logger.info(f"Processing heatmap for patient: {patient}, stain: {folder}, layer: {layer}")
                    spatial_info_dict = self._create_spatial_info_dict(patient_data, layer_score, folder)
                    canvas, att_img = self._create_canvas_and_att_img(spatial_info_dict, folder)

                    self.plot_patch_heatmap(patient, patient_dir, canvas, att_img, folder, str_label, fold,
                                            spatial_info_dict, self.args.patch_size, self.args.path_to_patches,
                                            heatmap_type=f"layer_{layer}")

    def _create_spatial_info_dict(self, patient_data, attention_scores, folder_id):
        coordinates = patient_data['metadata']['coordinates']
        filenames = patient_data['metadata']['filenames']

        return {filename[0]: ({'row1': int(coord.split(', ')[0][1:]),
                               'row2': int(coord.split(', ')[1]),
                               'col1': int(coord.split(', ')[2]),
                               'col2': int(coord.split(', ')[3][:-1])},
                              score)
                for filename, coord, score in zip(filenames, coordinates, attention_scores)
                if folder_id in filename[0]}


    def _create_canvas_and_att_img(self, spatial_info_dict, filename):
        max_x = max(info[0]['row2'] for info in spatial_info_dict.values())
        max_y = max(info[0]['col2'] for info in spatial_info_dict.values())

        canvas = np.zeros((max_x + self.args.patch_size, max_y + self.args.patch_size, 3), dtype=np.uint8)
        att_img = np.zeros((max_x + self.args.patch_size, max_y + self.args.patch_size), dtype=np.float64)

        for patch_name, (coordinates, score) in spatial_info_dict.items():
            y1, y2, x1, x2  = coordinates['row1'], coordinates['row2'], coordinates['col1'], coordinates['col2']

            att_img[y1:y2, x1:x2] = score[1]

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


    def _get_extreme_patches(self, spatial_info_dict, patch_size, path_to_patches, img_name, n=5):
        """Get the n highest and lowest attention score patches"""
        sorted_patches = sorted(spatial_info_dict.items(), key=lambda x: x[1][1][1])
        lowest_patches = sorted_patches[:n]
        highest_patches = sorted_patches[-n:]

        extreme_patches = []
        for patch_name, (coordinates, score) in lowest_patches + highest_patches:
            patch_image_path = os.path.normpath(os.path.join(path_to_patches, img_name, patch_name))
            if os.path.exists(patch_image_path):
                try:
                    patch_image = np.array(Image.open(patch_image_path))
                    if len(patch_image.shape) == 3 and patch_image.shape[2] == 4:
                        patch_image = patch_image[:, :, :3]
                    extreme_patches.append((patch_image, score))
                except Exception as e:
                    print(f"Error processing extreme patch {patch_image_path}: {str(e)}")

        return extreme_patches[:n], extreme_patches[n:]


    def _plot_patch_with_border(self, ax, patch, score, normalize_fn, border_width=4):
        """Helper function to plot a patch with a colored border"""
        height, width = patch.shape[:2]

        # Convert border color to uint8 range
        border_color = (np.array(plt.cm.jet(normalize_fn(score)))[:3] * 255).astype(np.uint8)

        # Create border by making a colored background larger than the patch
        border_patch = np.full((height + 2 * border_width, width + 2 * border_width, 3), border_color, dtype=np.uint8)

        # Ensure patch is uint8
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)

        # Place the original patch in the center
        border_patch[border_width:-border_width, border_width:-border_width] = patch

        # Display the combined image
        ax.imshow(border_patch)
        ax.axis('off')


    def plot_patch_heatmap(self, patient, output_dir, canvas, att_img, filename, label, fold, spatial_info_dict,
                           patch_size, path_to_patches, heatmap_type):
        output_folder = os.path.join(output_dir, "heatmaps")
        os.makedirs(output_folder, exist_ok=True)

        # Get extreme patches
        lowest_patches, highest_patches = self._get_extreme_patches(spatial_info_dict, patch_size, path_to_patches, filename)

        reconstructed_image_pil = Image.fromarray(canvas.astype(np.uint8), 'RGB')

        # Create figure with proper spacing - now with 2 columns instead of 3
        fig = plt.figure(figsize=(60, 30))  # Adjusted width for 2 panels
        gs = plt.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.02, wspace=0.02)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Whole Slide Image', size=60, pad=20)
        ax1.imshow(reconstructed_image_pil)
        ax1.axis('off')

        # Patch-based heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Patch-based heatmap', size=60, pad=20)
        ax2.imshow(reconstructed_image_pil)
        im2 = ax2.imshow(att_img, cmap='jet', alpha=0.6, vmin=0, vmax=np.max(att_img))
        ax2.axis('off')

        # Ensure both subplots have the same scale
        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.set_xlim(ax1.get_xlim())
            ax.set_ylim(ax1.get_ylim())

        # Create subplot for patches spanning both columns
        ax_patches = fig.add_subplot(gs[1, :])
        ax_patches.axis('off')

        # Calculate patch positions
        n_patches = len(lowest_patches)
        total_width = 0.9
        gap = 0.1
        patch_width = (total_width - gap) / (2 * n_patches)

        def plot_patches(patches, start_x, normalize_fn, title):
            for i, (patch, patch_score) in enumerate(patches):
                score = patch_score[1]
                x = start_x + i * patch_width
                box = ax_patches.inset_axes([x, 0.1, patch_width * 0.8, 0.8])
                self._plot_patch_with_border(box, patch, score, normalize_fn, border_width=8)

            # Add title below the patches
            mid_x = start_x + (n_patches * patch_width) / 2
            ax_patches.text(mid_x, -0.1, title,
                            ha='center', va='top',
                            fontsize=40,
                            transform=ax_patches.transAxes)

        # Plot patches with colored boxes
        max_score = np.max(att_img)
        score_normalize = lambda x: x / max_score

        # Plot lowest and highest scoring patches
        plot_patches(lowest_patches, 0.05, score_normalize, "Lowest Attention Score")
        plot_patches(highest_patches, 0.55, score_normalize, "Highest Attention Score")

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_ticks([])

        plt.suptitle(f'{filename} {label} - {heatmap_type}', size=60, y=0.95)

        output_path = os.path.normpath(os.path.join(output_folder, f"{filename}_heatmap_fold_{fold}_{heatmap_type}.png"))
        plt.savefig(output_path, dpi=fig.dpi, bbox_inches='tight')
        plt.close()
