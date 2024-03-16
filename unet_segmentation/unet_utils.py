# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:19:11 2024

@author: AmayaGS

"""

# Misc
import os
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hellslide
import openslide as osi

# PIL
from PIL import ImageFile, Image

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Patchification
from empatches_mod import EMPatches
emp = EMPatches()

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import gc
gc.enable()




def batch_generator(items, batch_size):
    count = 1
    chunk = []

    for item in items:
        if count % batch_size:
            chunk.append(item)
        else:
            chunk.append(item)
            yield chunk
            chunk = []
        count += 1

    if len(chunk):
        yield chunk


class patches_loader(Dataset):

    def __init__(self, image_dir, results_dir, transform, slide_level, patchsize, overlap):

        self.image_dir = image_dir
        self.results_dir = results_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.slide_level = slide_level
        self.patchsize = patchsize
        self.overlap = overlap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        img_name = self.images[index]
        file_type = img_name.split(".")[-1]
        len_file_type = len(file_type) + 1
        slide = osi.OpenSlide(img_path)
        properties = slide.properties
        results_folder_name = os.path.join(self.results_dir, img_name[:-len_file_type])

        if not os.path.exists(results_folder_name):

            if properties['openslide.objective-power'] == '40': # 40x is the default max magnification
                image = np.array(slide.read_region((0, 0), self.slide_level, slide.level_dimensions[self.slide_level]).convert('RGB'))

            elif properties['openslide.objective-power'] == '20':
                adjusted_level = int(self.slide_level + np.log2(int(properties['openslide.objective-power']) / 40)) # if max 20x, adjust level by +1
                image = np.array(slide.read_region((0, 0), adjusted_level, slide.level_dimensions[adjusted_level]).convert('RGB'))

            elif properties['openslide.objective-power'] == '10':
                adjusted_level = int(self.slide_level + np.log2(int(properties['openslide.objective-power']) / 40) * 2) # if max 10x, adjust level by +2
                image = np.array(slide.read_region((0, 0), adjusted_level, slide.level_dimensions[adjusted_level]).convert('RGB'))

            else:
                print(f"Slide {img_name} max magnification level is {properties['openslide.objective-power']}")
                image = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert('RGB'))

            img_patches, img_indices = emp.extract_patches(image, patchsize=self.patchsize, overlap=self.overlap)

            if self.transform:
                img_patches = [self.transform(img) for img in img_patches]

            return img_patches, img_indices, self.images[index][:-len_file_type]

        else:
            del self.images[index]
            return self.__getitem__(index)


def slide_loader(image_dir, results_dir, transform, slide_level, patchsize, overlap, slide_batch, num_workers, pin_memory,shuffle):

    dataset = patches_loader(image_dir=image_dir, results_dir=results_dir, transform=transform, slide_level=slide_level, patchsize=patchsize, overlap=overlap)
    slide_loader = DataLoader(dataset, batch_size=slide_batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    return slide_loader


def create_mask_and_patches(loader, model, batch_size, mean, std, device, path_to_save_mask_and_df, path_to_save_patches, coverage, keep_patches, patient_id_parsing):

    filename = path_to_save_mask_and_df + "/extracted_patches.csv"

    with open(filename, "a") as file:
        fileEmpty = os.stat(filename).st_size == 0
        headers  = ['Patient_ID', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        loop = tqdm(loader)
        model.eval()

        with torch.no_grad():

            for batch_idx, (img_patches, img_indices, label) in enumerate(loop):

                name = label[0] # check here what the parsing will be for your image name
                patient_id = eval(patient_id_parsing[0])
                num_patches = len(img_patches)

                print(f"Processing WSI: {name}, with {num_patches} patches")

                pred1 = []

                for i, batch in enumerate(batch_generator(img_patches, batch_size)):
                    batch = np.squeeze(torch.stack(batch), axis=1)
                    batch = batch.to(device=DEVICE, dtype=torch.float)

                    p1 = model(batch)
                    p1 = (p1 > 0.5) * 1
                    p1= p1.detach().cpu()
                    pred_patch_array = np.squeeze(p1)

                    for b in pred_patch_array:
                        pred1.append(b)

                    if keep_patches:

                        for patch in range(len(pred_patch_array)):
                            white_pixels = np.count_nonzero(pred_patch_array[patch])

                            if (white_pixels / len(pred_patch_array[patch])**2) > coverage:

                                patch_image = batch[patch].detach().cpu().numpy().transpose(1, 2, 0)

                                patch_image[:, :, 0] = (patch_image[:, :, 0] * std[0] + mean[0]).clip(0, 1)
                                patch_image[:, :, 1] = (patch_image[:, :, 1] * std[1] + mean[1]).clip(0, 1)
                                patch_image[:, :, 2] = (patch_image[:, :, 2] * std[2] + mean[2]).clip(0, 1)

                                patch_loc_array = np.array(torch.cat(img_indices[i*batch_size + patch]))
                                patch_loc_str = f"_x={patch_loc_array[0]}_x+1={patch_loc_array[1]}_y={patch_loc_array[2]}_y+1={patch_loc_array[3]}"
                                patch_name = name + patch_loc_str + ".png"
                                folder_location = os.path.join(path_to_save_patches, name)
                                os.makedirs(folder_location, exist_ok=True)
                                file_location = folder_location + "/" + patch_name
                                plt.imsave(file_location, patch_image)

                                data = {
                                        'Patient_ID': patient_id,
                                        'Filename': name,
                                        'Patch_name': patch_name,
                                        'Patch_coordinates': patch_loc_array,
                                        'File_location': file_location
                                        }

                                writer.writerow(data)

                    del p1, batch, pred_patch_array
                    gc.collect()

                merged_pre = emp.merge_patches(pred1, img_indices, rgb=False)
                plt.imsave(os.path.join(path_to_save_mask_and_df, "/binary_mask/" + name +".png"), merged_pre)

                del merged_pre, pred1, img_indices, img_patches
                gc.collect()

        writer.close