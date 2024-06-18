# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:54:34 2024

@author: AmayaGS
"""

# Misc
import os

# UTILS
from utils.utils_tissue_segmentation import save_patches

def tissue_segmentation(args):

    # Loading paths
    os.makedirs(args.directory, exist_ok =True)

    save_patches(image_dir= args.input_directory,
                 output_dir= args.directory,
                 slide_level= args.slide_level,
                 mask_level= args.mask_level,
                 patch_size= args.patch_size,
                 unet= args.unet,
                 unet_weights= args.unet_weights,
                 batch_size= args.patch_batch_size,
                 coverage= args.coverage,
                 name_parsing= args.name_parsing,
                 multistain= args.multistain)

# if __name__ == "__main__":
#     args = arg_parse()
#     args.input_directory = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_slides"
#     args.ouput_directory = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_results"
#     args.slide_level = 1
#     args.mask_level = 1
#     args.batch_size = 10
#     args.coverage = 0.3
#     args.unet = False
#     args.name_parsing = 'img_name.split("_")'
#     args.multistain = True
#     args.unet_weights = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"
#     main(args)