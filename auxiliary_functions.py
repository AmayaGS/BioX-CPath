# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:03:24 2023

@author: AmayaGS

"""

# Misc
import os
import random
import numpy as np

# PyTorch
import torch
from torch import Tensor

# PyG
from torch_geometric.utils import scatter


class Accuracy_Logger(object):

    """Accuracy logger"""
    def __init__(self, n_classes):
        #super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Define collate function
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def global_var_pool(x, batch, size= None):

    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        return x.var(dim=dim, keepdim=x.dim() <= 2)
    return scatter(x, batch, dim=dim, dim_size=size)