# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:27:20 2024

@author: AmayaGS
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import openslide as osi

from PIL import ImageFile
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from empatches_mod import EMPatches
emp = EMPatches()

from Models import UNet_512
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import gc
gc.enable()

#%%

class Data_Generator(Dataset):

    def __init__(self, image_dir, results_dir, transform, slide_level, patchsize, overlap):

        self.image_dir = image_dir
        self.results_dir = results_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.slide_level = slide_level

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        img_name = self.images[index]
        slide = osi.OpenSlide(img_path)
        properties = slide.properties

        results_folder_name = os.path.join(self.results_dir, img_name[:-5])
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
                image = np.array(slide.read_region((0, 0), self.slide_level, slide.level_dimensions[self.slide_level]).convert('RGB'))

            img_patches, img_indices = emp.extract_patches(image, patchsize=patchsize, overlap=overlap)

            if self.transform:
                img_patches = [self.transform(img) for img in img_patches]

            return img_patches, img_indices, self.images[index][:-4]

        else:
            del self.images[index]
            return self.__getitem__(index)


def slide_dataloader(image_dir, results_dir, transform, slide_level, patchsize, overlap, slide_batch, num_workers, pin_memory,shuffle):

    dataset = Data_Generator(image_dir=image_dir, results_dir=results_dir, transform=transform,slide_level=slide_level,patchsize=patchsize, overlap=overlap)
    slide_loader = DataLoader(dataset, batch_size=slide_batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    return slide_loader


#%%

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

def create_mask_and_patches(test_loader, model, batch_size, mean, std, device, path_to_save_mask_and_df, path_to_save_patches, coverage):

    loop = tqdm(test_loader)
    model.eval()
    with torch.no_grad():

        f = open(path_to_save_mask_and_df + "extracted_patches.csv", "a+")
        f.write("Patient_ID,Stain,Filename,Patch_name,Patch_coordinates,File_location\n")

        for batch_idx, (img_patches, img_indices, label) in enumerate(loop):

            name = label[0] # check here what the parsing will be for your image name
            patient_ID = label[0].split("_")[0]
            stain = label[0].split("_")[1] # this will work for a multi-stain dataset and should be removed otherwise
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
                            # I added the following 3 lines to fix the problem :ValueError: Floating point image RGB values must be in the 0..1 range.
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

                            f.write("{},{},{},{},{},{}\n".format(patient_ID,stain,name,patch_name,patch_loc_array,file_location))

                del p1, batch, pred_patch_array
                gc.collect()

            merged_pre = emp.merge_patches(pred1, img_indices, rgb=False)
            plt.imsave(os.path.join(path_to_save_mask_and_df, name +".png"), merged_pre)

            del merged_pre, pred1, img_indices, img_patches
            gc.collect()

        f.close

#%%

#### Results Paths #####
path_to_save_mask_and_df = r"D:\R4RA_patches\results/"
path_to_save_patches = r"D:\R4RA_patches/"

#%%

#### Loading Paths #####
test_img_dir = r"D:\R4RA_slides/"
path_to_checkpoints = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"

#%%

# Variables
NUM_WORKERS = 0
PIN_MEMORY = False
patchsize = 224
overlap = 0
shuffle = False
keep_patches = True
coverage = 0.3
slide_batch = 1
batch_size = 10
slide_level = 2
mean = (0.8946, 0.8659, 0.8638)
std = (0.1050, 0.1188, 0.1180)

#%%

test_transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

test_loader = slide_dataloader(test_img_dir, path_to_save_patches, test_transform, slide_level ,patchsize, overlap, slide_batch, NUM_WORKERS, PIN_MEMORY, shuffle)
print(len(test_loader)) ### this shoud be = Total_images/ batch size'''
#a = next(iter(test_loader)) # This is to test

#%%

Model = UNet_512().to(device=DEVICE,dtype=torch.float)
checkpoint = torch.load(path_to_checkpoints, map_location=DEVICE)
Model.load_state_dict(checkpoint['state_dict'], strict=True)
create_mask_and_patches(test_loader, Model, batch_size, mean, std, DEVICE, path_to_save_mask_and_df, path_to_save_patches, coverage)


#%%