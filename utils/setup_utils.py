# Misc
import os
import logging
import argparse
import ast
import random
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset
from torch import Tensor

# PyG
from torch_geometric.utils import scatter

from models.MUSTANG_model import MUSTANG_Classifier
from models.BioXCPath_model import BioXCPath_Classifier
from models.DeepGraphConv_model import DeepGraphConv
from models.patchGCN_model import PatchGCN
from models.GTP_model import GTP_Classifier
from models.TransMIL_model import TransMIL
from models.CLAM_model import GatedAttention as CLAM


MODEL_CONFIGS = {
    'BioXCPath': {
        'model_class': BioXCPath_Classifier,
        'use_args': True
    },
    'MUSTANG': {
        'model_class': MUSTANG_Classifier,
        'graph_mode': 'knn',
        'convolution': 'GAT',
        'encoding_size': 0,
        'heads': 2,
        'pooling_ratio': 0.7,
        'use_attention': False,
        'dropout': 0,
        'use_args': False
    },
    'CLAM': {
        'graph_mode': 'embedding',
        'convolution': 'Linear',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': CLAM,
        'use_attention': False,
        'dropout': 0.5,
        'use_args': False
    },
    'TransMIL': {
        'graph_mode': 'embedding',
        'convolution': 'Nystrom',
        'encoding_size': 0,
        'heads': 8,
        'pooling_ratio': 0,
        'model_class': TransMIL,
        'use_attention': False,
        'dropout': 0.5,
        'use_args': False
    },
    'PatchGCN': {
        'graph_mode': 'rag',
        'convolution': 'GCN',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': PatchGCN,
        'use_attention': False,
        'dropout': 0.5,
        'use_args': False
    },
    'DeepGraphConv': {
        'graph_mode': 'knn',
        'convolution': 'GIN',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': DeepGraphConv,
        'use_attention': False,
        'dropout': 0.5,
        'use_args': False
    },
    'GTP': {
        'graph_mode': 'rag',
        'convolution': 'ViT',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': GTP_Classifier,
        'use_attention': False,
        'dropout': 0.5,
        'use_args': False
    }
}


def get_model_config(args):
    config = MODEL_CONFIGS[args.model_name].copy()
    if config['use_args']:
        # For BioXCPath, use the args directly
        config['graph_mode'] = args.graph_mode
        config['convolution'] = args.convolution
        config['encoding_size'] = args.encoding_size
        config['heads'] = args.heads
        config['pooling_ratio'] = args.pooling_ratio
        config['use_attention'] = args.use_attention
        config['dropout'] = args.dropout
    return config


def setup_results_and_logging(args, log_type):
    current_directory = args.directory
    config = get_model_config(args)

    run_results_folder = (
        rf"{args.model_name}_{config['graph_mode']}_{config['convolution']}_PE_{config['encoding_size']}"
        f"_{args.embedding_net}_{args.dataset_name}_{args.seed}_{config['heads']}_{config['pooling_ratio']}"
        f"_{args.learning_rate}_{args.stain_type}_SAL_{config['use_attention']}_2dr_{config['dropout']}_2LL")

    results_dir = os.path.join(current_directory, "results", run_results_folder)
    os.makedirs(results_dir, exist_ok=True)

    log_file_path = results_dir + "/" + f"{run_results_folder}_{log_type}.log"

    # Create a new logger with a unique name
    logger = logging.getLogger(f'MUSTANG_{run_results_folder}_{log_type}')

    # Reset handlers to avoid duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M')

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return results_dir, logger

def parse_dict(string):
    try:
        return ast.literal_eval(string)
    except:
        raise argparse.ArgumentTypeError("Invalid dictionary format")


def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def global_var_pool(x, batch, size= None):

    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        return x.var(dim=dim, keepdim=x.dim() <= 2)
    return scatter(x, batch, dim=dim, dim_size=size)