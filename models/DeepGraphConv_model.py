# -*- coding: utf-8 -*-
"""
code from Patch-GCN - https://github.com/mahmoodlab/Patch-GCN

"""

from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures

from collections import OrderedDict
from os.path import join
import math
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class DeepGraphConv(torch.nn.Module):
    def __init__(self,
                 num_features=1024, hidden_dim=256,  dropout=0.25, n_classes=2):
        super(DeepGraphConv, self).__init__()

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def relocate(self):
        from torch_geometric.nn import DataParallel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to('cuda:0')
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to('cuda:0')
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, data, label):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))

        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector

        logits = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        Y_prob = torch.sigmoid(logits)
        results_dir = {'logits': logits, 'Y_prob': Y_prob, 'A_path': A_path}

        return logits, Y_prob, results_dir, label