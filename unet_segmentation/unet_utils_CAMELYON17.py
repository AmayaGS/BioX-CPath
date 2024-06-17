
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:19:11 2024

@author: AmayaGS

"""

# Misc
import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

# Pyvips
vipsbin = r'C:\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips


def fetch(region, patch_size, x, y):
    return region.crop(patch_size * y, patch_size * x, patch_size, patch_size)


def crop(image, patch_size, x, y):
    crop = image[patch_size * x: patch_size * x + patch_size, patch_size * y: patch_size * y + patch_size]
    x1 = patch_size * x
    x2 = patch_size * x + patch_size
    y1 = patch_size * y
    y2 = patch_size * y + patch_size
    return crop, [x1, x2, y1, y2]


def CAMELYON17_create_binary_mask(img_path, mask_path, mask_level, len_file_type):

        image = pyvips.Image.openslideload(img_path, level=mask_level, access='sequential')[0:3]
        np_image = np.asarray(image, dtype=np.uint8)
        im_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        binary_mask.dtype=np.uint8

        cv2.imwrite(mask_path, binary_mask)

        return binary_mask

# %%

def CAMELYON17_save_patches(image_dir, results_dir, mask_dir, slide_level, mask_level, patch_size, overlap, coverage, id_parsing):

    filename = results_dir + "/extracted_patches.csv"
    patches_dir = os.path.join(results_dir, 'patches')

    with open(filename, "a") as file:
        fileEmpty = os.stat(filename).st_size == 0
        headers  = ['Patient_ID', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        images = sorted(os.listdir(image_dir))

        for img in images:

            if img != 'patient_103_node_1.tif':

                img_path = os.path.join(image_dir, img)
                file_type = img.split(".")[-1]
                len_file_type = len(file_type) + 1
                img_name = img[:-len_file_type]
                patient_id = img_name.split('_')[0] + '_' + img_name.split('_')[1]
                mask_name = img_name + "_mask.png"
                mask_path = os.path.join(mask_dir, mask_name)

                if not os.path.exists(mask_path):

                    mask = CAMELYON17_create_binary_mask(img_path, mask_path, mask_level, len_file_type)

                else:

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                image = pyvips.Image.openslideload(img_path, level=slide_level, access='sequential')[0:3]
                mask = cv2.resize(mask, (image.width, image.height))

                print(f"Processing WSI: {img_name}, height: {image.height}, width: {image.width}")

                n_across = image.width // patch_size
                n_down = image.height // patch_size

                for x in range(0, n_down):

                    for y in range(0, n_across):

                        mask_crop = crop(mask, patch_size, x, y)
                        mask_patch = mask_crop[0]
                        coords = mask_crop[1]
                        white_pixels = np.count_nonzero(mask_patch)

                        if (white_pixels / len(mask_patch) ** 2) > coverage:

                            img_patch = fetch(image, patch_size, x, y)
                            patch_image = np.asarray(img_patch)

                            patch_loc_str = f"_x={coords[0]}_x+1={coords[1]}_y={coords[2]}_y+1={coords[3]}"
                            patch_name = img_name + patch_loc_str + ".png"
                            folder_location = os.path.join(patches_dir, img_name)
                            os.makedirs(folder_location, exist_ok=True)
                            file_location = folder_location + "/" + patch_name
                            plt.imsave(file_location, patch_image)

                            data = {
                                    'Patient_ID': patient_id,
                                    'Filename': img_name,
                                    'Patch_name': patch_name,
                                    'Patch_coordinates': coords,
                                    'File_location': file_location
                                    }

                            writer.writerow(data)

            else:
                continue

# %%