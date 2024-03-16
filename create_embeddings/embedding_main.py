# -*- coding: utf-8 -*-

"""
Created on Wed Feb 28 19:45:09 2024

@author: AmayaGS
"""

# Misc
import os
import pandas as pd
import pickle
import argparse

# sklearn
from sklearn.model_selection import StratifiedShuffleSplit

# PyTorch
import torch
from torchvision import transforms

# KRAG functions
from loaders import Loaders
from embedding_net import VGG_embedding, contrastive_resnet18, convNext
from embedding_utils import seed_everything, collate_fn_none, create_stratified_splits, create_embedding_graphs

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")



def arg_parse():

    parser = argparse.ArgumentParser(description="Feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag].")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'LUAD', 'LSCC'], help="Dataset name")
    parser.add_argument("--directory", type=str, default="/data/scratch/wpw030/KRAG", help="Location of patient label df and extracted patches df. Embeddings and graphs dictionaries will be kept here.")
    parser.add_argument("--label", type=str, default='Pathotype binary', help="Name of the target label in the metadata file")
    parser.add_argument("--patient_id", type=str, default='Patient_ID', help="Name of column containing the patient ID")
    parser.add_argument("--K", type=int, default=7, help="Number of nearest neighbours in k-NNG created from WSI embeddings")
    parser.add_argument("--embedding_vector_size", type=int, default=1000, help="Embedding vector size")
    parser.add_argument("--stratified_splits", type=int, default=10, help="Number of random stratified splits")
    parser.add_argument("--embedding_net", type=str, default="vgg16", choices=['resnet18', 'vgg16', 'convnext'], help="feature extraction network used")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--graph_mode", type=str, default="krag", choices=['knn', 'spatial', 'krag'], help="Change type of graph used for training here")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--slide_batch", type=int, default=1, help="Slide batch size - default 1")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")


    return parser.parse_args()


def main(args):

    # Set seed
    seed_everything(args.seed)

    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load df with patient_id and corresponding labels here, to merge with extracted patches.
    patient_labels = pd.read_csv(args.directory + "/patient_labels.csv")
    # Load file with all extracted patches metadata and locations.
    extracted_patches = pd.read_csv(args.directory + "/extracted_patches.csv")

    df = pd.merge(extracted_patches, patient_labels, on= args.patient_id)
    # Drop duplicates to obtain the actuals patient IDs that have a label assigned by the pathologist
    df_labels = df.drop_duplicates(subset= args.patient_id)
    ids = list(df_labels[args.patient_id])

    sss_dict_name = args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl"
    if not os.path.exists(sss_dict_name):
        # create the dictionary containing the patient ID dictionary of the stratified random splits
        create_stratified_splits(extracted_patches, patient_labels, args.patient_id, args.label, args.train_fraction, args.seed, args.dataset_name)

    # Create dictionary with patient ID as key and Dataloaders containing the corresponding patches as values.
    slides = Loaders().slides_dataloader(df, ids, transform, slide_batch= args.slide_batch, num_workers= args.num_workers, shuffle= False, collate= collate_fn_none, label= args.label, patient_id= args.patient_id)

    if args.embedding_net == 'resnet18':
        # Load weights for resnet18
        embedding_net = contrastive_resnet18(args.directory + '/tenpercent_resnet18.pt')
    elif args.embedding_net == 'vgg16':
        # Load weights for vgg16
        embedding_net = VGG_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'convnext':
        # Load weights for convnext
        embedding_net = convNext()

    if use_gpu:
         embedding_net.cuda()

    print(f"Start creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")
    embedding_dict, knn_dict, rag_dict, krag_dict = create_embedding_graphs(embedding_net, slides, k=args.K, include_self=True)
    print(f"Done creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")

    with open(args.directory + f"/embedding_dict_{args.dataset_name}_{args.embedding_net}.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)  # encode dict into Pickle
        print("Done writing embedding_dict into pickle file")

    with open(args.directory + f"/knn_dict_{args.dataset_name}_{args.embedding_net}.pkl", "wb") as file:
        pickle.dump(knn_dict, file)  # encode dict into Pickle
        print("Done writing knn_dict into pickle file")

    with open(args.directory + f"/rag_dict_{args.dataset_name}_{args.embedding_net}.pkl", "wb") as file:
        pickle.dump(rag_dict, file)  # encode dict into Pickle
        print("Done writing rag_dict into pickle file")

    with open(args.directory + f"/krag_dict_{args.dataset_name}_{args.embedding_net}.pkl", "wb") as file:
        pickle.dump(krag_dict, file)  # encode dict into Pickle
        print("Done writing krag_dict into pickle file")



if __name__ == "__main__":
    args = arg_parse()
    args.directory = r"C:\Users\Amaya\Documents\PhD\MUSTANGv2\min_code_krag\data"
    args.label = 'Pathotype binary'
    args.patient_id = 'Patient_ID'
    args.K = 8
    args.dataset_name = "RA"
    args.embedding_net = 'vgg16'
    main(args)