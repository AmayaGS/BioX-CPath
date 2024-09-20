# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:52:19 2024

@author: AmayaGS
"""

# Misc
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Hellslide
import openslide as osi

# PyTorch
import torch
from torchvision import transforms

# UNET model
from models.UNet_models import UNet_512

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if DEVICE == torch.device('cuda'):
    print("Using CUDA")


mean = [0.8946, 0.8659, 0.8638]
std = [0.1050, 0.1188, 0.1180]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


def crop(image, patch_size, row, col):

    row1 = patch_size * row
    row2 = patch_size * row + patch_size
    col1 = patch_size * col
    col2 = patch_size * col + patch_size

    crop = image[row1: row2, col1: col2]
    coordinates = [row1, row2, col1, col2]

    return crop, coordinates


def openslide_crop(slide, patch_size, downsample, slide_level, row, col):

    row1 = patch_size * row
    row2 = patch_size * row + patch_size
    col1 = patch_size * col
    col2 = patch_size * col + patch_size

    crop = np.asarray(slide.read_region((col1 * downsample, row1 * downsample), slide_level, (patch_size, patch_size)).convert('RGB'))
    coordinates = [row1, row2, col1, col2]

    return crop, coordinates


def tresh_binary_mask(slide, mask_path, contours_path, mask_level):

    image = np.array(slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert('RGB'), dtype=np.uint8)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (10,5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask.dtype = np.uint8

    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask1 = np.zeros_like(binary_mask)

    contour_results = image.copy()
    for cont in contours:
        area=cv2.contourArea(cont)
        if area>500:
            cv2.drawContours(contour_results, [cont], -1, (0,0,255), 3)
            cv2.drawContours(mask1, [cont], -1, (255,255,255), -1)
    cv2.imwrite(mask_path, mask1)
    cv2.imwrite(contours_path,  cv2.cvtColor(contour_results, cv2.COLOR_RGB2BGR))

    return mask1


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


def unet_binary_mask(model, transform, slide, mask_path, contours_path, mask_level, batch_size, patch_size):

    image = np.array(slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert('RGB'), dtype=np.uint8)
    width = slide.level_dimensions[mask_level][0]
    height = slide.level_dimensions[mask_level ][1]
    downsample = int(slide.level_downsamples[mask_level])
    binary_mask = np.zeros((height, width))

    n_across = width // patch_size
    n_down = height // patch_size

    count = 1
    batch = []
    batch_coords = []

    for row in range(0, n_down):
        for col in range(0, n_across):

            img_crop = openslide_crop(slide, patch_size, downsample, mask_level, row, col)
            patch = img_crop[0]
            coords = img_crop[1]

            if count < batch_size:
                batch.append(patch)
                batch_coords.append(coords)
                count += 1
            else:
                batch.append(patch)
                batch_coords.append(coords)

                T_batch = [transform(np.array(img)) for img in batch]
                T_batch = np.squeeze(torch.stack(T_batch), axis=1)
                T_batch = T_batch.to(device=DEVICE, dtype=torch.float)

                p1 = model(T_batch)
                p1 = (p1 > 0.5) * 1
                p1 = np.squeeze(p1.detach().cpu())
                pred = [b for b in p1]

                for (pred_mask, img, coords) in zip(pred, batch, batch_coords):

                    # populating binary mask
                    row1, row2, col1, col2 = coords[0], coords[1], coords[2], coords[3]
                    binary_mask[row1:row2, col1:col2] = pred_mask

                batch = []
                batch_coords = []
                count = 1

    img = binary_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(binary_mask)

    contour_results = image.copy()
    for cont in contours:
        area=cv2.contourArea(cont)
        if area>500:
            cv2.drawContours(contour_results, [cont], -1, (0,0,255), 3)
            cv2.drawContours(mask, [cont], -1, (255,255,255), -1)
    cv2.imwrite(contours_path, contour_results)
    plt.imsave(mask_path, binary_mask)

    return binary_mask


def save_patches(image_dir,
                 output_dir,
                 slide_level,
                 mask_level,
                 patch_size,
                 unet,
                 unet_weights,
                 batch_size,
                 coverage,
                 name_parsing,
                 stain_parsing,
                 logger):

    mask_dir = os.path.join(output_dir, 'masks')
    contours_dir = os.path.join(output_dir, 'contours')
    thumbnails_dir = os.path.join(output_dir, 'thumbnails')
    results_dir = os.path.join(output_dir, f'extracted_patches_{slide_level}')
    patches_dir = os.path.join(results_dir, 'patches')
    filename = os.path.join(results_dir, 'extracted_patches.csv')

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(contours_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    # Load existing extracted_patches.csv if it exists
    if os.path.exists(filename):
        existing_patches = pd.read_csv(filename)
        processed_images = set(existing_patches['Filename'].unique())
    else:
        processed_images = set()

    with open(filename, "a") as file:
        fileEmpty = os.stat(filename).st_size == 0
        headers = ['Patient_ID', 'Stain_type', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)

        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        if unet:
            model = UNet_512().to(device=DEVICE, dtype=torch.float)
            checkpoint = torch.load(unet_weights, map_location=DEVICE)
            model.load_state_dict(checkpoint['state_dict'], strict=True)

        images = sorted(os.listdir(image_dir))

        for img in images:
            # if img != 'patient_103_node_1.tif': # special case for camelyon17 -
            # image is corrupted  TODO add exception if image is corrupted? probably needs to be caught downstream
            img_path = os.path.join(image_dir, img)
            file_type = img.split(".")[-1]
            len_file_type = len(file_type) + 1
            img_name = img[:-len_file_type]

            # Skip if this image has already been processed
            if img_name in processed_images:
                logger.info(f"Skipping {img_name} as it has already been processed.")
                continue
            # patient ID
            try:
                patient_id = eval(name_parsing)
            except Exception as e:
                logger.info(f"Error parsing patient ID for {img}: {str(e)}. "
                      f"Check the name_parsing argument. Skipping this image for now.")
                continue
            # stain type
            try:
                stain_type = eval(stain_parsing)
            except Exception as e:
                logger.info(f"Error parsing stain type for {img}: {str(e)}. Using 'default'.")
                stain_type = "default"

            mask_name = img_name + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            contour_path = os.path.join(contours_dir, mask_name)
            thumbnail_path = os.path.join(thumbnails_dir, img_name)

            slide = osi.OpenSlide(img_path)
            width = slide.level_dimensions[slide_level][0]
            height = slide.level_dimensions[slide_level][1]
            downsample = int(slide.level_downsamples[slide_level])
            lowest_level = slide.level_count - 1

            logger.info(f"Processing WSI: {img_name}, height: {height}, width: {width}, slide level: {slide_level}, "
                  f"downsample factor: {downsample}")

            if not os.path.exists(thumbnail_path + '.png'):

                thumbnail = np.asarray(slide.read_region((0, 0), lowest_level, (slide.level_dimensions[lowest_level][0], slide.level_dimensions[lowest_level][1])).convert('RGB'))
                plt.imsave(thumbnail_path + '.png', thumbnail)

            if os.path.exists(mask_path):
                logger.info("Loading pre-existing mask.")

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # TODO add tifffile reader here maybe
                # in case the image is actually in grayscale, pass it to 0 and 1
                mask = cv2.resize(mask, (width, height)) # resize to slide level dimensions
                _,mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            if not os.path.exists(mask_path) and not unet:
                logger.info("No mask on file. Creating adaptive binary thresholding mask.")

                mask = tresh_binary_mask(slide, mask_path, contour_path, mask_level)
                mask = cv2.resize(mask, (width, height))

            if not os.path.exists(mask_path) and unet:
                logger.info("No mask on file. Creating unet binary mask.")

                mask = unet_binary_mask(model, transform, slide, mask_path, contour_path, mask_level, batch_size, patch_size)
                mask = cv2.resize(mask, (width, height))

            n_across = width // patch_size
            n_down = height // patch_size

            count = 1
            batch = []
            batch_coords = []

            for row in range(0, n_down):
                for col in range(0, n_across):

                    mask_crop = crop(mask, patch_size, row, col)
                    patch = mask_crop[0]
                    coords = mask_crop[1]

                    if count < batch_size:
                        batch.append(patch)
                        batch_coords.append(coords)
                        count += 1
                    else:
                        batch.append(patch)
                        batch_coords.append(coords)

                    for (pred_mask, coords) in zip(batch, batch_coords):
                        white_pixels = np.count_nonzero(pred_mask)

                        if (white_pixels / patch_size ** 2) > coverage:
                            img = np.asarray(slide.read_region((coords[2] * downsample, coords[0] * downsample),
                                                               slide_level, (patch_size, patch_size)).convert('RGB'))

                            patch_loc_str = f"_row1={coords[0]}_row2={coords[1]}_col1={coords[2]}_col2={coords[3]}"
                            patch_name = img_name + patch_loc_str + ".png"
                            folder_location = os.path.join(patches_dir, img_name)
                            os.makedirs(folder_location, exist_ok=True)
                            file_location = folder_location + "/" + patch_name
                            plt.imsave(file_location, img)

                            data = {
                            'Patient_ID': patient_id,
                            'Filename': img_name,
                            'Stain_type': stain_type,
                            'Patch_name': patch_name,
                            'Patch_coordinates': coords,
                            'File_location': file_location
                            }

                            writer.writerow(data)

                            batch = []
                            batch_coords = []
                            count = 1

            processed_images.add(img_name)

    logger.info("Done processing all images.")