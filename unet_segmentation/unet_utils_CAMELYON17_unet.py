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

# # Hellslide
# import openslide as osi



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



# Misc
import os
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pyvips
vipsbin = r'C:\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips

# TIFFF
import tifffile as tifff



def fetch(region, patch_size, x, y):
    return region.crop(patch_size * y, patch_size * x, patch_size, patch_size)

def crop(image, patch_size, x, y):
    crop = image[patch_size * x: patch_size * x + patch_size, patch_size * y: patch_size * y + patch_size]
    x1 = patch_size * x
    x2 = patch_size * x + patch_size
    y1 = patch_size * y
    y2 = patch_size * y + patch_size
    return crop, [x1, x2, y1, y2]


def CAMELYON16_save_patches(image_dir, mask_dir, results_dir, slide_level, patchsize, overlap, patch_size, path_to_save_mask_and_df, path_to_save_patches, coverage):


    filename = path_to_save_mask_and_df + "/extracted_patches.csv"

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
            mask_name = img_name.split(".")[0] + "_mask.tif"
            mask_path = os.path.join(mask_dir, mask_name)
            file_type = img_name.split(".")[-1]
            len_file_type = len(file_type) + 1
            patient_id = img_name.split(".")[0]

            results_folder_name = os.path.join(results_dir, img_name[:-len_file_type])

            if not os.path.exists(results_folder_name):

                image = pyvips.Image.openslideload(img_path, level=slide_level, access='sequential')[0:3]
                mask = cv2.imread(mask_path)
                mask = cv2.resize(mask, (image.height, image.width))

                print(f"Processing WSI: {patient_id}, height: {image.height}, width: {image.width}")

                n_across = image.width // patch_size
                n_down = image.height // patch_size

                file_location = os.path.join(path_to_save_binary_mask, patient_id)
                plt.imsave(file_location + 'thresh.png' , binary_mask)

                n_across = image.width // patch_size
                n_down = image.height // patch_size

                for x in range(0, n_down):
                    #print("row {} ...".format(x))

                    for y in range(0, n_across):

                        mask_crop = crop(mask, patch_size, x, y)
                        mask_patch = mask_crop[0]
                        coords = mask_crop[1]
                        white_pixels = np.count_nonzero(mask_patch)
                        #print(white_pixels, white_pixels / len(mask_patch) ** 2)

                        if (white_pixels / len(mask_patch) ** 2) > coverage:

                            img_patch = fetch(image, patch_size, x, y)
                            patch_image = np.asarray(img_patch)

                            patch_loc_str = f"_x={coords[0]}_x+1={coords[1]}_y={coords[2]}_y+1={coords[3]}"
                            patch_name = patient_id + patch_loc_str + ".png"
                            folder_location = os.path.join(path_to_save_patches, patient_id)
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

    writer.close

# %%

# Misc
import os
import numpy as np
import cv2
import csv

# Pyvips
vipsbin = r'C:\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips

# TIFFF
import tifffile as tifff

img_path = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\images"
list_img = os.listdir(img_path)

path_to_save_patches = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\data\patches"
path_to_save_binary_mask = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\data\binary_masks"
slide_level = 2
patch_size = 224
coverage = 0.4

# # UNET model
# from unet_models import UNet_512

# path_to_checkpoints = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"

# mean = [0.8946, 0.8659, 0.8638]
# std = [0.1050, 0.1188, 0.1180]

# #mean = [0.9, 0.9, 0.9]
# #std = [0.1, 0.1, 0.1]

# model = UNet_512().to(device=DEVICE, dtype=torch.float)
# checkpoint = torch.load(path_to_checkpoints, map_location=DEVICE)
# model.load_state_dict(checkpoint['state_dict'], strict=True)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)])


# %%

def crop(img, patch_size, x, y):
    # for some reason pyvips crop use col (y) * row (x)
    crop = img.crop(patch_size * y, patch_size * x, patch_size, patch_size)
    x1 = patch_size * x
    x2 = patch_size * x + patch_size
    y1 = patch_size * y
    y2 = patch_size * y + patch_size
    return crop, [x1, x2, y1, y2]

# %%

#mask = tifff.imread(mask_path, key=slide_level)

#patient_id = img_name.split(".")[0]

#print(f"Processing WSI: {patient_id}, height: {image.height}, width: {image.width}")

# filename = path_to_save_mask_and_df + "/extracted_patches.csv"

# with open(filename, "a") as file:
#     fileEmpty = os.stat(filename).st_size == 0
#     headers  = ['Patient_ID', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
#     writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
#     if fileEmpty:
#         writer.writeheader()  # file doesn't exist yet, write a header

#     images = sorted(os.listdir(image_dir))

#     for index in range(len(images)):

#         img_path = os.path.join(image_dir, images[index])
#         img_name = images[index]
#         mask_name = img_name.split(".")[0] + "_mask.tif"
#         mask_path = os.path.join(mask_dir, mask_name)
#         file_type = img_name.split(".")[-1]
#         len_file_type = len(file_type) + 1

#         results_folder_name = os.path.join(results_dir, img_name[:-len_file_type])

#         if not os.path.exists(results_folder_name):

#kernel = np.ones((3, 3))

for img in list_img:

    path = os.path.join(img_path, img)
    patient_id = path.split("\\")[-1][:-4]

    image = pyvips.Image.openslideload(path, level=3, access='sequential')[0:3]
    np_image = np.asarray(image, dtype=np.uint8)
    im_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    file_location = os.path.join(path_to_save_binary_mask, patient_id)
    plt.imsave(file_location + 'thresh.png' , binary_mask)


# %%

for img in list_img:

    path = os.path.join(img_path, img)

    image = pyvips.Image.openslideload(path, level=3, access='sequential')[0:3]
    np_image = np.asarray(image, dtype=np.uint8)

    n_across = image.width // patch_size
    n_down = image.height // patch_size
    max_x = image.height
    max_y = image.width

    binary_mask = np.zeros((max_x, max_y))

    batch_size = 10
    count = 1
    batch = []
    batch_coords = []

    patient_id = path.split("\\")[-1][:-4]

    model.eval()
    #with torch.no_grad():

    for x in range(0, n_down):

        for y in range(0, n_across):

            img_patch = crop(image, patch_size, x, y)
            np_img_patch = np.asarray(img_patch[0], dtype=np.int32)
            coords = img_patch[1]

            if count < batch_size:
                batch.append(np_img_patch)
                batch_coords.append(coords)
                count += 1
            else:
                batch.append(np_img_patch)
                batch_coords.append(coords)

                # T_batch = [transform(img) for img in batch]
                # T_batch = np.squeeze(torch.stack(T_batch), axis=1)
                # T_batch = T_batch.to(device=DEVICE, dtype=torch.float)

                # p1 = model(T_batch)
                # p1 = (p1 > 0.5) * 1
                # p1 = np.squeeze(p1.detach().cpu())
                # pred = [b for b in p1]

                for i, (mask, img, coords) in enumerate(zip(pred, batch, batch_coords)):

                    # populating binary mask
                    x1, x2, y1, y2 = coords[0], coords[1], coords[2], coords[3]
                    binary_mask[x1:x2, y1:y2] = mask

                    # saving patches if binary mask coverage is beyond chosen threshold
                    white_pixels = np.count_nonzero(mask)

                    if (white_pixels / patch_size ** 2) > coverage:

                        patch_loc_str = f"_x={coords[0]}_x+1={coords[1]}_y={coords[2]}_y+1={coords[3]}"
                        patch_name = patient_id + patch_loc_str + ".png"
                        folder_location = os.path.join(path_to_save_patches, patient_id)
                        os.makedirs(folder_location, exist_ok=True)
                        file_location = folder_location + "/" + patch_name
                        #os.makedirs(file_location, exist_ok=True)
                        #plt.imsave(file_location, img)

                batch = []
                batch_coords = []
                count = 1

    file_location = os.path.join(path_to_save_binary_mask, patient_id)
    plt.imsave(file_location + '.png' , binary_mask)

# %%

# Add a few cells of distance around the tumor.

image_spacing = mask_image.getSpacing()[0]
image_downsampling = mask_image.getLevelDownsample(level=level)
image_level_spacing = image_spacing * image_downsampling
distance_threshold_pixels = dilation_distance / (image_level_spacing * 2.0)
image_binary_array = np.less(image_distance_array, distance_threshold_pixels)

mask_image.close()

# Fill the holes in the dilated tumor mask and label the regions.
#
image_filled_array = scipy.ndimage.morphology.binary_fill_holes(input=image_binary_array)
image_evaluation_mask = skimage.measure.label(input=image_filled_array, connectivity=2)

# %%

image = pyvips.Image.openslideload(path, level=3, access='sequential')[0:3]
np_image = np.asarray(image, dtype=np.uint8)


# %%

import cv2

#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%


im_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(im_gray, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

_, thresh4 = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

thresh2 =cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

thresh3 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

plt.imshow(thresh4)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#cv2.imshow('thresh', thresh)

# %%

cnts, hier = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(thresh3, cnts, 0, 255, cv2.CV_FILLED)

# Find contour with the maximum area
c = max(cnts, key=cv2.contourArea)

res = np.zeros_like(im_gray)  # Create new zeros images for storing the result.

# Fill the contour with white color - draw the filled contour on res image.
res = cv2.drawContours(res, [c], -1, 255, -1)

# Compute the center of the contour
# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
M = cv2.moments(c)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# Use floodFill for filling the center of the contour
cv2.floodFill(thresh3, 255)


from skimage import color
from skimage import morphology
from skimage.io import imread

image_filled_array = morphology.binary_erosion(thresh3)
image_evaluation_mask = skimage.measure.label(input=image_filled_array, connectivity=2)


# Creating kernel
kernel = np.ones((3, 3), np.uint8)

# Using cv2.erode() method
image = cv2.erode(thresh3, kernel)


# %%

def CAMELYON17_create_binary_mask(image_dir, slide_level, path_to_save_binary_mask):

        images = sorted(os.listdir(image_dir))

        for img in images:

            img_path = os.path.join(image_dir, img)
            img_name = img[:-4]
            mask_name = img_name + "_mask.png"
            mask_path = os.path.join(path_to_save_binary_mask, mask_name)

            image = pyvips.Image.openslideload(img_path, level=slide_level, access='sequential')[0:3]
            np_image = np.asarray(image, dtype=np.uint8)
            im_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            binary_mask = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
            binary_mask.dtype=np.uint8

            cv2.imwrite(mask_path, binary_mask)

# %%
# %%

image_dir = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\images"
results_dir = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\data\patches"
mask_dir = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON17\data\binary_masks"
slide_level = 2
patch_size = 224
overlap = 0
coverage = 0.3

#CAMELYON17_create_binary_mask(image_dir, slide_level, path_to_save_binary_mask)

# %%

# Misc
import os
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pyvips
vipsbin = r'C:\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips

# TIFFF
import tifffile as tifff



def fetch(region, patch_size, x, y):
    return region.crop(patch_size * y, patch_size * x, patch_size, patch_size)

def crop(image, patch_size, x, y):
    crop = image[patch_size * x: patch_size * x + patch_size, patch_size * y: patch_size * y + patch_size]
    x1 = patch_size * x
    x2 = patch_size * x + patch_size
    y1 = patch_size * y
    y2 = patch_size * y + patch_size
    return crop, [x1, x2, y1, y2]



def CAMELYON17_save_patches(image_dir, results_dir, mask_dir, slide_level, patch_size, overlap, coverage):

    filename = results_dir + "/extracted_patches.csv"

    with open(filename, "a") as file:
        fileEmpty = os.stat(filename).st_size == 0
        headers  = ['Patient_ID', 'Filename', 'Patch_name', 'Patch_coordinates', 'File_location']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        images = sorted(os.listdir(image_dir))

        for img in images:

            img_path = os.path.join(image_dir, img)
            file_type = img.split(".")[-1]
            len_file_type = len(file_type) + 1
            img_name = img[:-len_file_type]
            mask_name = img_name + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)

            results_folder_name = os.path.join(results_dir, img_name)

            image = pyvips.Image.openslideload(img_path, level=slide_level, access='sequential')[0:3]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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
                        folder_location = os.path.join(results_dir, img_name)
                        os.makedirs(folder_location, exist_ok=True)
                        file_location = folder_location + "/" + patch_name
                        plt.imsave(file_location, patch_image)

                        data = {
                                'Patient_ID': img_name,
                                'Filename': img_name,
                                'Patch_name': patch_name,
                                'Patch_coordinates': coords,
                                'File_location': file_location
                                }

                        writer.writerow(data)

# %%

CAMELYON17_save_patches(image_dir, results_dir, mask_dir, slide_level, patch_size, overlap, coverage)