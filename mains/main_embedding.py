# Misc
import os
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
from models.embedding_models import VGG_embedding, resnet18_embedding, resnet50_embedding, convNext
from models.embedding_models import contrastive_resnet18
from models.embedding_models import GigaPath_embedding, UNI_embedding, BiOptimus_embedding, Phikon_embedding
from utils.embedding_utils import seed_everything, collate_fn_none, create_embedding_graphs, save_graph_statistics

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = "cuda"


def patch_embedding(args, logger):

    # Set seed
    seed_everything(args.seed)

    # ImageNet transforms - good for UNI, GigaPath, Phikon
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ]
    )


    if args.embedding_net == 'BiOptimus':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.707223, 0.578729, 0.703617],
                                    std=[0.211883, 0.230117, 0.177517])
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

    # Create dictionary with patient ID as key and Dataloaders containing the corresponding patches as values.
    slides = Loaders().slides_dataloader(df, ids, transform, slide_batch= args.slide_batch, num_workers= args.num_workers, shuffle= False, collate= collate_fn_none, label= args.label, patient_id= args.patient_id)

    if args.embedding_net == 'resnet18':
        # Load weights for resnet18
        embedding_net = resnet18_embedding(embedding_vector_size=args.embedding_vector_size)
    if args.embedding_net == 'ssl_resnet18':
        # Load weights for pretrained resnet18
        weight_path = os.path.join(args.embedding_weights, "Ciga", "tenpercent_resnet18.pt")
        embedding_net = contrastive_resnet18(weight_path, embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'resnet50':
        # Load weights for resnet 50
        embedding_net = resnet50_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'ssl_resnet50':
        # Load weights for resnet 50
        embedding_net = resnet50_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'vgg16':
        # Load weights for vgg16
        embedding_net = VGG_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'convnext':
        # Load weights for convnext
        embedding_net = convNext(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'CTransPath':
        # Load weights for CTransPath
        weight_path = os.path.join(args.embedding_weights, "CTransPath", "ctranspath.pth")
        embedding_net = CTransPath_embedding(weight_path, embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'Lunit':
        # Load weights for Lunit
        embedding_net = Lunit_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'GigaPath':
        # Load weights for GigaPath
        embedding_net = GigaPath_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'Phikon':
        # Load weights for Phikon
        embedding_net = Phikon_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'BiOptimus':
        # Load weights for BioOptimus
        embedding_net = BiOptimus_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'UNI':
        # Load weights for UNI
        embedding_net = UNI_embedding(embedding_vector_size=args.embedding_vector_size)

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
