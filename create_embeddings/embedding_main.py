# -*- coding: utf-8 -*-

"""
Created on Wed Feb 28 19:45:09 2024

@author: AmayaGS
"""

# Misc
import os
import pandas as pd
import pickle

# sklearn

# PyTorch
import torch
from torchvision import transforms

# KRAG functions
from utils.utils_dataloaders import Loaders
from models.embedding_models import VGG_embedding, resnet18_embedding, contrastive_resnet18, resnet50_embedding, convNext
from utils.embedding_utils import seed_everything, collate_fn_none, create_stratified_splits, create_embedding_graphs

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()

def patch_embedding(args):

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
    extracted_patches = pd.read_csv(args.directory + "/results_" + str(args.slide_level) + "/extracted_patches.csv")

    df = pd.merge(extracted_patches, patient_labels, on= args.patient_id)

    if args.multistain:
        df = df[df['Stain_type'] == args.stain_type]
        df_labels = df.drop_duplicates(subset= args.patient_id)
        ids = list(df_labels[args.patient_id])
    else:
      # Drop duplicates to obtain the actuals patient IDs that have a label assigned by the pathologist
        df_labels = df.drop_duplicates(subset= args.patient_id)
        ids = list(df_labels[args.patient_id])

    sss_dict_name = args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl"
    if not os.path.exists(sss_dict_name):
        # create the dictionary containing the patient ID dictionary of the stratified random splits
        create_stratified_splits(extracted_patches, patient_labels, args.patient_id, args.label, args.train_fraction, args.val_fraction, args.stratified_splits, args.seed, args.dataset_name, args.directory)

    # Create dictionary with patient ID as key and Dataloaders containing the corresponding patches as values.
    slides = Loaders().slides_dataloader(df, ids, transform, slide_batch= args.slide_batch, num_workers= args.num_workers, shuffle= False, collate= collate_fn_none, label= args.label, patient_id= args.patient_id, multistain= args.multistain)

    if args.embedding_net == 'resnet18':
        # Load weights for resnet18
        embedding_net = resnet18_embedding()
    if args.embedding_net == 'ssl_resnet18':
        # Load weights for resnet18
        embedding_net = contrastive_resnet18('/data/scratch/wpw030/MUSTANGv2_scratch/tenpercent_resnet18.pt')
    elif args.embedding_net == 'resnet50':
        # Load weights for convnext
        embedding_net = resnet50_embedding()
    elif args.embedding_net == 'vgg16':
        # Load weights for vgg16
        embedding_net = VGG_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'convnext':
        # Load weights for convnext
        embedding_net = convNext()

    if use_gpu:
         embedding_net.cuda()

    print(f"Start creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")
    embedding_dict, knn_dict, rag_dict, krag_dict = create_embedding_graphs(embedding_net, slides, k=args.K, include_self=True, multistain=args.multistain)
    print(f"Done creating {args.dataset_name} embeddings and graph dictionaries for {args.embedding_net}")

    dictionaries = os.path.join(args.directory, "dictionaries")
    os.makedirs(dictionaries, exist_ok = True)

    with open(dictionaries + f"/embedding_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)  # encode dict into Pickle
        print("Done writing embedding_dict into pickle file")

    with open(dictionaries + f"/knn_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(knn_dict, file)  # encode dict into Pickle
        print("Done writing knn_dict into pickle file")

    with open(dictionaries + f"/rag_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(rag_dict, file)  # encode dict into Pickle
        print("Done writing rag_dict into pickle file")

    with open(dictionaries + f"/krag_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(krag_dict, file)  # encode dict into Pickle
        print("Done writing krag_dict into pickle file")


#
# if __name__ == "__main__":
#
#   args = arg_parse()
#   args.directory = "/data/scratch/wpw030/CAMELYON16/results_5/"
#   args.label = 'label'
#   args.patient_id = 'Patient_ID'
#   args.K = 8
#   args.dataset_name = "CAMELYON16"
#   args.embedding_net = 'resnet18'
#   args.multistain = False
#   args.stain_type = "H&E"
#   main(args)