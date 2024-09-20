# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:03:24 2023

@author: AmayaGS

"""

# Misc
import os
import logging
import argparse
import ast

import random
import numpy as np

from PIL import Image

# PyTorch
import torch
from torch.utils.data import Dataset
from torch import Tensor

# PyG
from torch_geometric.utils import scatter

from models.KRAG_model import KRAG_Classifier
from models.MUSTANG_model import MUSTANG_Classifier
from models.DeepGraphConv_model import DeepGraphConv
from models.patchGCN_model import PatchGCN
from models.GTP_model import GTP_Classifier
from models.TransMIL_model import TransMIL
from models.CLAM_model import GatedAttention as CLAM


MODEL_CONFIGS = {
    'KRAG': {
        'graph_mode': 'krag',
        'model_class': KRAG_Classifier,
        'use_args': True
    },
    'MUSTANG': {
        'graph_mode': 'krag',
        'model_class': MUSTANG_Classifier,
        'use_args': True
    },
    'CLAM': {
        'graph_mode': 'embedding',
        'convolution': 'Linear',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': CLAM,
        'use_args': False
    },
    'TransMIL': {
        'graph_mode': 'embedding',
        'convolution': 'Nystrom',
        'encoding_size': 0,
        'heads': 8,
        'pooling_ratio': 0,
        'model_class': TransMIL,
        'use_args': False
    },
    'PatchGCN': {
        'graph_mode': 'rag',
        'convolution': 'GCN',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': PatchGCN,
        'use_args': False
    },
    'DeepGraphConv': {
        'graph_mode': 'knn',
        'convolution': 'GIN',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': DeepGraphConv,
        'use_args': False
    },
    'GTP': {
        'graph_mode': 'rag',
        'convolution': 'ViT',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': GTP_Classifier,
        'use_args': False
    }
}


def get_model_config(args):
    config = MODEL_CONFIGS[args.model_name].copy()
    if config['use_args']:
        # For KRAG, use the args directly
        config['graph_mode'] = args.graph_mode
        config['convolution'] = args.convolution
        config['encoding_size'] = args.encoding_size
        config['heads'] = args.heads
        config['pooling_ratio'] = args.pooling_ratio
    return config


def setup_results_and_logging(args, log_type):
    current_directory = args.directory
    config = get_model_config(args)

    run_results_folder = (
        f"{args.model_name}_{config['graph_mode']}_{config['convolution']}_PE_{config['encoding_size']}"
        f"_{args.embedding_net}_{args.dataset_name}_{args.seed}_{config['heads']}_{config['pooling_ratio']}"
        f"_{args.learning_rate}_{args.scheduler}_{args.stain_type}_L1_{args.l1_norm}")

    results_dir = os.path.join(current_directory, "results", run_results_folder)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(results_dir, run_results_folder + log_type + ".log")

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger('KRAG')

    return results_dir, logger


def parse_dict(string):
    try:
        return ast.literal_eval(string)
    except:
        raise argparse.ArgumentTypeError("Invalid dictionary format")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def global_var_pool(x, batch, size= None):

    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        return x.var(dim=dim, keepdim=x.dim() <= 2)
    return scatter(x, batch, dim=dim, dim_size=size)