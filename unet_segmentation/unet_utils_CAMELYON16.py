# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:56:02 2024

@author: AmayaGS
"""

# Misc
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Hellslide
import openslide as osi

# TIFFF
#import tifffile as tifff
import cv2
#import skimage as ski
#from skimage.transform import resize

def crop(image, patch_size, row, col):

    row1 = patch_size * row
    row2 = patch_size * row + patch_size
    col1 = patch_size * col
    col2 = patch_size * col + patch_size

    crop = image[row1: row2, col1: col2]

    return crop, [row1, row2, col1, col2]


def CAMELYON16_save_patches(image_dir, mask_dir, results_dir, slide_level, patch_size, coverage):

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
            patient_id = img_name.split(".")[0]
            mask_name = patient_id + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            image = osi.OpenSlide(img_path)
            #mask = tifff.imread(mask_path, key=slide_level)

            width = image.level_dimensions[slide_level][0]
            height = image.level_dimensions[slide_level][1]
            downsample = int(image.level_downsamples[slide_level])

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _,mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.resize(mask, (width, height))
            #mask = ski.io.imread(mask_path)
            #resize(mask, (height, width)).shape
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
                        plt.imsave(file_location, patch_image)

                        data = {
                                'Patient_ID': patient_id,
                                'Filename': img_name,
                                'Patch_name': patch_name,
                                'Patch_coordinates': coords,
                                'File_location': file_location
                                }

                        writer.writerow(data)