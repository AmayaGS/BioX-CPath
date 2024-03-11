
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

# %%

#### Reconstruct Patches ####

#### Resutls Paths #####
path_to_save_mask_and_df = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches\results/"
path_to_save_patches = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches/"

#%%

#### Loading Paths #####
test_img_dir = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_slides/"
path_to_checkpoints = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"

path_df_UNet = path_to_save_mask_and_df + "\extracted_patches.csv"
df_unet = pd.read_csv(path_df_UNet)
df_unet.set_index(['File_location'],inplace=True)
path_to_patches = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches\LOUV-R4RA-L976_CD20_PATH - 2021-02-12 15.53.39"

#%%

spatial_info_dict = {}

for index, row in df_unet.iterrows():
    # Extracting coordinates from the df
    coordinates = list(map(int, row['Patch_coordinates'].strip('[]').split()))


    patch_name = f"{index}"

    # Save the patch name and coordinates to the dictionary
    spatial_info_dict[patch_name] = {
        'x1': coordinates[2],
        'x2': coordinates[3],
        'y1': coordinates[0],
        'y2': coordinates[1]
    }

#%%

max_x = int(max(spatial_info_dict[filename]['x2'] for filename in spatial_info_dict))
max_y = int(max(spatial_info_dict[filename]['y2'] for filename in spatial_info_dict))


canvas = np.zeros((max_y + 224, max_x + 224, 3), dtype=np.uint8)

# Place each patch at its specified location on the canvas
for filename in spatial_info_dict:
    x1, x2, y1, y2 = spatial_info_dict[filename]['x1'], spatial_info_dict[filename]['x2'], spatial_info_dict[filename]['y1'], spatial_info_dict[filename]['y2']


    patch_image_path = os.path.join(path_to_patches, filename)
    patch_image = np.array(Image.open(patch_image_path))
    if patch_image.shape[2] == 4:
        patch_image = patch_image[:, :, :3]

    canvas[y1:y2, x1:x2, :] = patch_image

reconstructed_image_pil = Image.fromarray(canvas)
reconstructed_image_pil.show()

# %%