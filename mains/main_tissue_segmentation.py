
# Misc
import os

# UTILS
from utils.tissue_segmentation_utils import save_patches

def tissue_segmentation(args, logger):

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
                 name_parsing= args.patient_ID_parsing,
                 stain_parsing=args.stain_parsing,
                 logger= logger)
