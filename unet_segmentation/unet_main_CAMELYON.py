# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:07:04 2024

@author: AmayaGS
"""

# Misc
import os
import argparse

from unet_utils_CAMELYON import CAMELYON_save_patches


def arg_parse():

    parser = argparse.ArgumentParser(description="Input arguments for unet segmentation and patching of Whole Slide Images")

    parser.add_argument('--input_directory', type=str, default= "/slides/", help='Input data directory')
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'LUAD', 'LSCC', 'CAMELYON16', 'CAMELYON17'], help="Dataset name")
    parser.add_argument('--patches_directory', type=str, default= "/patches/", help='Results directory path')
    parser.add_argument('--mask_directory', type=str, default="masks/", help='Mask directory path')
    parser.add_argument('--results_directory', type=str, default= "/data/", help='Results directory path')
    parser.add_argument('--patchsize', type=int, default=224, help='Patch size (default: 224)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap (default: 0)')
    parser.add_argument('--coverage', type=float, default=0.3, help='Coverage (default: 0.3)')
    parser.add_argument('--slide_level', type=int, default=1, help='Slide level (default: 2)')

    args = parser.parse_args()

    return args

def main(args):

    # Loading Paths
    os.makedirs(args.patches_directory, exist_ok = True)
    os.makedirs(args.results_directory, exist_ok =True)

    CAMELYON_save_patches(args.dataset_name, args.input_directory, args.mask_directory, args.results_directory, args.slide_level, args.patchsize, args.overlap, args.patchsize, args.results_directory, args.patches_directory, args.coverage)

# %%

if __name__ == "__main__":
    args = arg_parse()
    args.dataset_name = 'CAMELYON16'
    args.input_directory = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON16\images"
    args.mask_directory = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON16\masks"
    args.patches_directory = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON16\data\patches"
    args.results_directory = r"C:\Users\Amaya\Documents\PhD\Data\Camelyon\CAMELYON16\data_v2"
    args.slide_level = 3
    args.coverage = 0.3
    main(args)

# %%