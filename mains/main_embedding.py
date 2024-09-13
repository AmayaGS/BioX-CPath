# -*- coding: utf-8 -*-

"""
Created on Wed Feb 28 19:45:09 2024

@author: AmayaGS
"""

# Misc
import os
from multiprocessing import process

import pandas as pd
import pickle

# for GigaPath and UNI patch encoder - make sure it's up to date
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Hugging Face stuff
from huggingface_hub import login, hf_hub_download

# PyTorch
import torch
from torchvision import transforms

# KRAG functions
from utils.dataloaders_utils import Loaders
from models.embedding_models import VGG_embedding, resnet18_embedding, contrastive_resnet18, resnet50_embedding, convNext
from utils.embedding_utils import seed_everything, collate_fn_none, create_stratified_splits, create_embedding_graphs, save_graph_statistics

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = "cuda"


def patch_embedding(args, logger):

    # Set seed
    seed_everything(args.seed)

    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.embedding_net == 'UNI':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    # Load df with patient_id and corresponding labels here, to merge with extracted patches.
    patient_labels = pd.read_csv(args.directory + "/patient_labels.csv")
    # Load file with all extracted patches metadata and locations.
    extracted_patches = pd.read_csv(args.directory + "/extracted_patches_" + str(args.slide_level) + "/extracted_patches.csv")

    df = pd.merge(extracted_patches, patient_labels, on= args.patient_id)

    # Drop duplicates to obtain the actuals patient IDs that have a label assigned by the pathologist
    df_labels = df.drop_duplicates(subset= args.patient_id)
    ids = list(df_labels[args.patient_id])

    sss_dict_name = args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl"
    if not os.path.exists(sss_dict_name):
        # create the dictionary containing the patient ID dictionary of the stratified random splits
        create_stratified_splits(extracted_patches, patient_labels, args.patient_id, args.label, args.train_fraction, args.val_fraction, args.stratified_splits, args.seed, args.dataset_name, args.directory)

    # Create dictionary with patient ID as key and Dataloaders containing the corresponding patches as values.
    slides = Loaders().slides_dataloader(df, ids, transform, slide_batch= args.slide_batch, num_workers= args.num_workers, shuffle= False, collate= collate_fn_none, label= args.label, patient_id= args.patient_id)

    if args.embedding_net == 'resnet18':
        # Load weights for resnet18
        embedding_net = resnet18_embedding(embedding_vector_size=args.embedding_vector_size)
    if args.embedding_net == 'ssl_resnet18':
        # Load weights for pretrained resnet18
        embedding_net = contrastive_resnet18(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'resnet50':
        # Load weights for resnet 50
        embedding_net = resnet50_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'vgg16':
        # Load weights for vgg16
        embedding_net = VGG_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'convnext':
        # Load weights for convnext
        embedding_net = convNext(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'gigapath':
        # Load weights for gigapath
        embedding_net = timm.create_model('resnet50', pretrained=True) #reemplace
    elif args.embedding_net == 'UNI':
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        # Load weights for uni
        local_dir = r"C:\Users\Amaya\Documents\PhD\Data\WSI_foundation\UNI_weights"
        embedding_net = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                                          init_values=1e-5, num_classes=0, dynamic_img_size=True)
        embedding_net.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"),
                                         map_location=device), strict=True)

    if use_gpu:
         embedding_net.cuda()

    logger.info(f"Start creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")
    embedding_dict, knn_dict, rag_dict, krag_dict, statistics = create_embedding_graphs(embedding_net, slides, k=args.K, include_self=True, stain_types=args.stain_types, edge_types=args.edge_types)
    logger.info(f"Done creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")

    save_graph_statistics(statistics, args.directory)

    dictionaries = os.path.join(args.directory, "dictionaries")
    os.makedirs(dictionaries, exist_ok = True)

    with open(dictionaries + f"/embedding_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)  # encode dict into Pickle
        logger.info("Done writing embedding_dict into pickle file")

    with open(dictionaries + f"/knn_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(knn_dict, file)  # encode dict into Pickle
        logger.info("Done writing knn_dict into pickle file")

    with open(dictionaries + f"/rag_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(rag_dict, file)  # encode dict into Pickle
        logger.info("Done writing rag_dict into pickle file")

    with open(dictionaries + f"/krag_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(krag_dict, file)  # encode dict into Pickle
        logger.info("Done writing krag_dict into pickle file")
