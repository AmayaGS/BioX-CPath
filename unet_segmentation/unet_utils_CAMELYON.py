# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:57:50 2024

@author: AmayaGS
"""

# Misc
import os
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# # Pyvips
# vipsbin = r'C:\vips-dev-8.15\bin'
# os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
# import pyvips

# Hellslide
import openslide as osi

# TIFFF
import tifffile as tifff

# OpenCV
import cv2



def CAMELYON_create_binary_mask(img_path, mask_path, mask_level, len_file_type):

        slide = osi.OpenSlide(img_path)
        np_image = np.asarray(slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert('RGB'), dtype=np.uint8)
        im_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        binary_mask.dtype=np.uint8

        cv2.imwrite(mask_path, binary_mask)

        return binary_mask

def get_mask_name(img_name, dataset_name):

    mask_extension = "_mask.png" if dataset_name == "CAMELYON16" else "_mask.tif"

    return img_name.split(".")[0] + mask_extension

def crop(image, patch_size, row, col):

    row1 = patch_size * row
    row2 = patch_size * row + patch_size
    col1 = patch_size * col
    col2 = patch_size * col + patch_size

    crop = image[row1: row2, col1: col2]

    return crop, [row1, row2, col1, col2]


def CAMELYON_save_patches(dataset_name, image_dir, mask_dir, results_dir, slide_level, mask_level, patch_size, overlap, coverage):


    filename = results_dir + "/extracted_patches.csv"
    patches_dir = os.path.join(results_dir, 'patches')


    with open(filename, "a") as file:
        fileEmpty = os.stat(filename).st_size == 0
        headers  = ['Patient_ID', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        images = sorted(os.listdir(image_dir))

        for index in range(len(images)):

            img_path = os.path.join(image_dir, images[index])
            img_name = images[index]
            mask_name = get_mask_name(img_name, dataset_img_name.split(".")[0] + ("_mask.png" if "CAMELYON16" in img_name else "_mask.tif")
            mask_path = os.path.join(mask_dir, mask_name)
            file_type = img_name.split(".")[-1]
            len_file_type = len(file_type) + 1

            results_folder_name = os.path.join(results_dir, img_name[:-len_file_type])

            if not os.path.exists(results_folder_name):

                image = osi.OpenSlide(img_path)
                mask = tifff.imread(mask_path, key=slide_level)

                patient_id = img_name.split(".")[0]

                width = image.level_dimensions[slide_level][0]
                height = image.level_dimensions[slide_level][1]
                downsample = int(image.level_downsamples[slide_level])

                print(f"Processing WSI: {patient_id}, height: {height}, width: {width}, downsample: {downsample}")

                n_across = width // patch_size
                n_down = height // patch_size

                for row in range(0, n_down):

                    for col in range(0, n_across):

                        mask_crop = crop(mask, patch_size, row, col)
                        mask_patch = mask_crop[0]
                        coords = mask_crop[1]
                        white_pixels = np.count_nonzero(mask_patch)

                        if (white_pixels / len(mask_patch) ** 2) > coverage:

                            patch_image = np.asarray(image.read_region((coords[2] * downsample, coords[0] * downsample), slide_level, (patch_size, patch_size)).convert('RGB'))

                            patch_loc_str = f"_row1={coords[0]}_row2={coords[1]}_col1={coords[2]}_col2={coords[3]}"
                            patch_name = patient_id + patch_loc_str + ".png"
                            folder_location = os.path.join(patches_dir, patient_id)
                            os.makedirs(folder_location, exist_ok=True)
                            file_location = folder_location + "/" + patch_name
                            #os.makedirs(file_location, exist_ok=True)
                            plt.imsave(file_location, patch_image)

                            data = {
                                    'Patient_ID': patient_id,
                                    'Filename': img_name,
                                    'Patch_name': patch_name,
                                    'Patch_coordinates': coords,
                                    'File_location': file_location
                                    }

                            writer.writerow(data)