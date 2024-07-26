# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:22:22 2023

@author: AmayaGS
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmaxp
from torch_geometric.nn import SAGPooling



class KRAG_Classifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, heads, pooling_ratio, walk_length, conv_type, attention):
        super(KRAG_Classifier, self).__init__()
        self.attention =  attention
        self.krag = pooling_network(in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type)

        if self.attention:
            self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
            nn.init.xavier_uniform_(self.attention_weights)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data, filenames):
        x, all_patches_per_layer, all_patches_cumulative = self.krag(data, filenames)

        if self.attention:
            attention_scores = torch.matmul(x, self.attention_weights) # 1* hidden_dim * 2
            attention_scores = F.softmax(attention_scores, dim= -1)
            x = torch.sum(x * attention_scores, dim=0).unsqueeze(0) # hidden_dim * 2

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out, all_patches_per_layer, all_patches_cumulative



class pooling_network(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type):
        super().__init__()
        self.heads = heads
        self.pooling_ratio = pooling_ratio
        self.walk_length = walk_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if conv_type == "GAT":

            self.conv1 = GATv2Conv(in_features + self.walk_length, hidden_dim, heads=self.heads, concat=False)
            self.conv2 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)
            self.conv3 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)
            self.conv4 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)

        if conv_type == "GCN":

            self.conv1 = GCNConv(in_features + self.walk_length, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim)

        if conv_type == "GraphSAGE":

            self.conv1 = SAGEConv(in_features + self.walk_length, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
            self.conv4 = SAGEConv(hidden_dim, hidden_dim)

        if conv_type == "GIN":

            self.conv1 = GINConv(nn.Sequential(Linear(in_features + self.walk_length, hidden_dim)))
            self.conv2 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))
            self.conv3 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))
            self.conv4 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))

        self.pool1 = SAGPooling(hidden_dim, self.pooling_ratio)
        self.pool2 = SAGPooling(hidden_dim, self.pooling_ratio)
        self.pool3 = SAGPooling(hidden_dim, self.pooling_ratio)
        self.pool4 = SAGPooling(hidden_dim, self.pooling_ratio)


    def forward(self, data, filenames):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.walk_length > 0:
            rwpe = data.random_walk_pe

        # Initialize attention attention_scores for each layer and a cumulative score
        attention_scores_per_layer = [torch.zeros(len(filenames), device=self.device) for _ in range(4)]
        cumulative_attention_scores = torch.zeros(len(filenames), device=self.device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[0] = self.update_and_normalize_attention_scores(attention_scores_per_layer[0], perm, score)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[1] = self.update_and_normalize_attention_scores(attention_scores_per_layer[1], perm, score)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[2] = self.update_and_normalize_attention_scores(attention_scores_per_layer[2], perm, score)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[3] = self.update_and_normalize_attention_scores(attention_scores_per_layer[3], perm, score)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score)

        # Normalize cumulative attention attention_scores
        normalized_cumulative_scores = self.normalize_attention_scores(cumulative_attention_scores)

        all_patches_per_layer = [
            [(filename[0], score.item()) for filename, score in zip(filenames, layer_scores.cpu())]
            for layer_scores in attention_scores_per_layer
        ]
        all_patches_cumulative = [(filename[0], score.item()) for filename, score in
                                  zip(filenames, normalized_cumulative_scores.cpu())]

        x = x1 + x2 + x3 + x4

        return x, all_patches_per_layer, all_patches_cumulative

    def update_and_normalize_attention_scores(self, current_scores, perm, new_scores):
        current_scores[perm] = new_scores.to(self.device)
        return self.normalize_attention_scores(current_scores)

    def update_cumulative_scores(self, cumulative_scores, perm, new_scores):
        cumulative_scores[perm] += new_scores.to(self.device)
        return cumulative_scores

    def normalize_attention_scores(self, scores):
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return torch.ones_like(scores, device=self.device) * 0.5  # If all attention_scores are the same, return 0.5

#
# class KRAG_Classifier_per_layer(nn.Module):
#     def __init__(self, in_features, hidden_dim, num_classes, heads, pooling_ratio, walk_length, conv_type, attention):
#
#         super(KRAG_Classifier_per_layer, self).__init__()
#
#         self.attention =  attention
#
#         self.krag = pooling_network_per_layer(in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type)
#
#         if self.attention:
#             self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
#             nn.init.xavier_uniform_(self.attention_weights)
#
#         self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
#         self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)
#
#     def forward(self, data, filename):
#         x, layer1, layer2, layer3, layer4 = self.krag(data, filename)
#
#         if self.attention:
#             attention_scores = torch.matmul(x, self.attention_weights) # 1* hidden_dim * 2
#             attention_scores = F.softmax(attention_scores, dim= -1)
#             x = torch.sum(x * attention_scores, dim=0).unsqueeze(0) # hidden_dim * 2
#
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.lin2(x)
#         x = F.relu(x)
#         x_logits = self.lin3(x)
#         x_out = F.softmax(x_logits, dim=1)
#
#         return x_logits, x_out, layer1, layer2, layer3, layer4
#
#
# class pooling_network_per_layer(torch.nn.Module):
#
#
#     """"""
#
#     def __init__(self, in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type):
#
#         super().__init__()
#
#         self.heads = heads
#         self.pooling_ratio = pooling_ratio
#         self.walk_length = walk_length
#
#         if conv_type == "GAT":
#
#             self.conv1 = GATv2Conv(in_features + self.walk_length, hidden_dim, heads=self.heads, concat=False)
#             self.conv2 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)
#             self.conv3 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)
#             self.conv4 = GATv2Conv(hidden_dim + self.walk_length, hidden_dim, heads=self.heads, concat=False)
#
#         if conv_type == "GCN":
#
#             self.conv1 = GCNConv(in_features + self.walk_length, hidden_dim)
#             self.conv2 = GCNConv(hidden_dim, hidden_dim)
#             self.conv3 = GCNConv(hidden_dim, hidden_dim)
#             self.conv4 = GCNConv(hidden_dim, hidden_dim)
#
#         if conv_type == "GraphSAGE":
#
#             self.conv1 = SAGEConv(in_features + self.walk_length, hidden_dim)
#             self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#             self.conv3 = SAGEConv(hidden_dim, hidden_dim)
#             self.conv4 = SAGEConv(hidden_dim, hidden_dim)
#
#         if conv_type == "GIN":
#
#             self.conv1 = GINConv(nn.Sequential(Linear(in_features + self.walk_length, hidden_dim)))
#             self.conv2 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))
#             self.conv3 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))
#             self.conv4 = GINConv(nn.Sequential(Linear(hidden_dim, hidden_dim)))
#
#         self.pool1 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool2 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool3 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool4 = SAGPooling(hidden_dim, self.pooling_ratio)
#
#
#     def forward(self, data, filenames):
#
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         if self.walk_length > 0:
#             rwpe = data.random_walk_pe
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         perm_1 = perm
#         score_1 = score
#         patches_1 = set(perm_1.tolist())
#         patches_with_attn  = [(filename, score_1[perm_1.tolist().index(i)].item()) for i, filename in enumerate(filenames) if i in patches_1]
#         patches_no_attn = [(filename, np.nan) for i, filename in enumerate(filenames) if i not in patches_1]
#         layer1 = patches_with_attn + patches_no_attn
#
#         if self.walk_length > 0:
#             rwpe = rwpe[perm]
#             x = torch.cat([x, rwpe], dim=1)
#
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, perm, score = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         perm_2 = perm
#         score_2 = score
#         patches_2 = set(perm_2.tolist())
#         patches_with_attn  = [(filename, score_2[perm_2.tolist().index(i)].item()) for i, filename in enumerate(filenames) if i in patches_2]
#         patches_no_attn = [(filename, np.nan) for i, filename in enumerate(filenames) if i not in patches_2]
#         layer2 = patches_with_attn + patches_no_attn
#
#         if self.walk_length > 0:
#             rwpe = rwpe[perm]
#             x = torch.cat([x, rwpe], dim=1)
#
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, perm, score = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         perm_3 = perm
#         score_3 = score
#         patches_3 = set(perm_3.tolist())
#         patches_with_attn  = [(filename, score_3[perm_3.tolist().index(i)].item()) for i, filename in enumerate(filenames) if i in patches_3]
#         patches_no_attn = [(filename, np.nan) for i, filename in enumerate(filenames) if i not in patches_3]
#         layer3 = patches_with_attn + patches_no_attn
#
#         if self.walk_length > 0:
#             rwpe = rwpe[perm]
#             x = torch.cat([x, rwpe], dim=1)
#
#         x = self.conv4(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, perm, score = self.pool4(x, edge_index, None, batch)
#         x4 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         perm_4 = perm
#         score_4 = score
#         patches_4 = set(perm_4.tolist())
#         patches_with_attn  = [(filename, score_4[perm_4.tolist().index(i)].item()) for i, filename in enumerate(filenames) if i in patches_4]
#         patches_no_attn = [(filename, np.nan) for i, filename in enumerate(filenames) if i not in patches_4]
#         layer4 = patches_with_attn + patches_no_attn
#
#         x = x1 + x2 + x3 + x4
#
#         return x, layer1, layer2, layer3, layer4