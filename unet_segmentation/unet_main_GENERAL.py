# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:54:34 2024

@author: AmayaGS
"""

# Misc
import os
import argparse

from unet_utils_GENERAL import save_patches


def arg_parse():

    parser = argparse.ArgumentParser(description="Input arguments for unet segmentation and patching of Whole Slide Images")

    parser.add_argument('--input_directory', type=str, default= "/slides/", help='Input data directory')
    parser.add_argument('--output_directory', type=str, default= "/output_dir/", help='Results directory path')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size (default: 224)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap (default: 0)')
    parser.add_argument('--coverage', type=float, default=0.3, help='Coverage (default: 0.3)')
    parser.add_argument('--slide_level', type=int, default=2, help='Slide level (default: 2)')
    parser.add_argument('--mask_level', type=int, default=3, help='Slide level (default: 3)')
    parser.add_argument('--unet', type=bool, default=False, help='Whether to create binary mask using UNet segmentation or binary thresholding')
    parser.add_argument('--unet_weights', type=str, default= "/path_to_unet_weights", help='Path to model checkpoints')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size (default: 10)')
    parser.add_argument('--name_parsing', type=str, default='img_name.split(".")[0]', help='String parsing to obtain patient ID from image filename')
    parser.add_argument('--multistain', type=bool, default=False, help='Whether the dataset contains multiple types of staining. Will generate a extracted_patches.csv with stain type info.')

    args = parser.parse_args()

    return args

def main(args):

    # Loading paths
    os.makedirs(args.ouput_directory, exist_ok =True)

    save_patches(image_dir= args.input_directory,
                 output_dir= args.ouput_directory,
                 slide_level= args.slide_level,
                 mask_level= args.mask_level,
                 patch_size= args.patch_size,
                 unet= args.unet,
                 unet_weights= args.unet_weights,
                 batch_size= args.batch_size,
                 coverage= args.coverage,
                 name_parsing= args.name_parsing,
                 multistain= args.multistain)


# %%

if __name__ == "__main__":
    args = arg_parse()
    args.input_directory = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_slides"
    args.ouput_directory = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_results"
    args.slide_level = 1
    args.mask_level = 1
    args.batch_size = 10
    args.coverage = 0.3
    args.unet = False
    args.name_parsing = 'img_name.split("_")'
    args.multistain =  True
    args.unet_weights = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"
    main(args)

# %%