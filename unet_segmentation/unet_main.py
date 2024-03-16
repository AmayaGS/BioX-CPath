# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:27:20 2024

@author: AmayaGS
"""

# Misc
import os
import argparse

# PIL
from PIL import ImageFile, Image

# PyTorch
import torch
from torchvision import transforms

# Patchification
from empatches_mod import EMPatches
emp = EMPatches()

# UNET model
from unet_models import UNet_512
from unet_utils import slide_loader, create_mask_and_patches

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import gc
gc.enable()



def arg_parse():

    parser = argparse.ArgumentParser(description="Input arguments for unet segmentation and patching of Whole Slide Images")

    parser.add_argument('--input_directory', type=str, default= "/slides/", help='Input data directory')
    parser.add_argument('--patches_directory', type=str, default= "/patches/", help='Results directory path')
    parser.add_argument('--results_directory', type=str, default= "/data/", help='Results directory path')
    parser.add_argument('--path_to_checkpoints', type=str, default=r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar", help='Path to model checkpoints')
    parser.add_argument('--patient_id_parsing', type=str, default='name.split("_")[0]', help='String parsing to obtain patient ID from image filename')
    parser.add_argument('--NUM_WORKERS', type=int, default=0, help='Number of workers (default: 0)')
    parser.add_argument('--PIN_MEMORY', type=bool, default=False, help='Pin memory (default: False)')
    parser.add_argument('--patchsize', type=int, default=224, help='Patch size (default: 224)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap (default: 0)')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle (default: False)')
    parser.add_argument('--keep_patches', type=bool, default=True, help='Keep patches (default: True)')
    parser.add_argument('--coverage', type=float, default=0.3, help='Coverage (default: 0.3)')
    parser.add_argument('--slide_batch', type=int, default=1, help='Slide batch (default: 1)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size (default: 10)')
    parser.add_argument('--slide_level', type=int, default=2, help='Slide level (default: 2)')
    parser.add_argument('--mean', nargs='+', type=float, default=[0.8946, 0.8659, 0.8638], help='Mean (default: [0.8946, 0.8659, 0.8638])')
    parser.add_argument('--std', nargs='+', type=float, default=[0.1050, 0.1188, 0.1180], help='Standard deviation (default: [0.1050, 0.1188, 0.1180])')

    args = parser.parse_args()

    return args


def main(args):

    # Loading Paths
    os.makedirs(args.patches_directory, exist_ok = True)
    os.makedirs(args.results_directory, exist_ok =True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)])

    loader = slide_loader(args.input_directory, args.patches_directory, transform, args.slide_level, args.patchsize, args.overlap,
                               args.slide_batch, args.NUM_WORKERS, args.PIN_MEMORY, args.shuffle)

    Model = UNet_512().to(device=DEVICE, dtype=torch.float)
    checkpoint = torch.load(args.path_to_checkpoints, map_location=DEVICE)
    Model.load_state_dict(checkpoint['state_dict'], strict=True)

    create_mask_and_patches(loader, Model, args.batch_size, args.mean, args.std, DEVICE, args.results_directory,
                            args.patches_directory, args.coverage, args.keep_patches, args.patient_id_parsing)



if __name__ == "__main__":
    args = arg_parse()
    args.input_directory = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_slides/"
    args.patches_directory = r"C:\Users\Amaya\Documents\PhD\MUSTANGv2\min_code_krag\data\patches"
    args.results_directory = r"C:\Users\Amaya\Documents\PhD\MUSTANGv2\min_code_krag\data"
    args.path_to_checkpoints =r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\IHC_segmentation\IHC_Synovium_Segmentation\UNet weights\UNet_512_1.pth.tar"
    args.patient_id_parsing = 'name.split("_")[0]'
    args.coverage = 0.3
    main(args)