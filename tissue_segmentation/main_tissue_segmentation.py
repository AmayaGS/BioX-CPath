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
                 stain_parsing=args.stain_parsing,
                 multistain= args.multistain)
