import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmaxp
from torch_geometric.nn import SAGPooling
from torch_geometric.data import Data


class KRAG_Classifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, heads, pooling_ratio, walk_length, conv_type, attention):
        super(KRAG_Classifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = attention
        self.krag = pooling_network(in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type)

        if self.attention:
            self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
            nn.init.xavier_uniform_(self.attention_weights)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data, filenames):
        data = data.to(self.device)
        x, all_patches_per_layer, all_patches_cumulative, layer_data = self.krag(data, filenames)

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

        return x_logits, x_out, all_patches_per_layer, all_patches_cumulative, layer_data



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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        original_node_attr = data.node_attr
        original_edge_attr = edge_attr.clone()
        original_filenames = [filename[0] for filename in filenames]

        layer_data = []

        if self.walk_length > 0:
            rwpe = data.random_walk_pe

        # Initialize attention patient_graphs for each layer and a cumulative score
        attention_scores_per_layer = [[None for _ in range(len(filenames))] for _ in range(4)]
        cumulative_attention_scores = [None for _ in range(len(filenames))]

        # Create initial node mapping
        node_idx_mapping = {i: i for i in range(x.size(0))} # k=node,v=original idx
        original_edge_mapping = {i: edge_index.t()[i].tolist() for i in range(edge_index.size(1))} # k=edge,v=original idx

        edge_index_mapping_rev = {str(v): k for k, v in original_edge_mapping.items()}

        # Layer 1
        x, (edge_index1, att_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, edge_attr, batch)
        att_weights = self.filter_edges(edge_index1, att_weights1, perm)

        node_idx_mapping, filenames, edge_idx_mapping, edge_old_idx_new_idx = self.update_node_mapping(node_idx_mapping, perm, original_filenames, edge_index_mapping_rev, edge_index)
        layer_data = self.layer_graph(layer_data, x, edge_index, edge_attr, score, original_node_attr,
                                      att_weights, filenames, node_idx_mapping,
                                      edge_index_mapping_rev, edge_old_idx_new_idx, original_edge_attr)
        x1 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[0] = self.update_attention_scores(attention_scores_per_layer[0], perm, score, node_idx_mapping)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score, node_idx_mapping)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        # Layer 2
        x, (edge_index2, att_weights2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, edge_attr, batch)
        att_weights = self.filter_edges(edge_index2, att_weights2, perm)

        node_idx_mapping, filenames, edge_idx_mapping, edge_old_idx_new_idx = self.update_node_mapping(node_idx_mapping, perm, original_filenames, edge_index_mapping_rev, edge_index)
        layer_data = self.layer_graph(layer_data, x, edge_index, edge_attr, score, original_node_attr,
                                      att_weights, filenames, node_idx_mapping, edge_index_mapping_rev, edge_old_idx_new_idx, original_edge_attr)
        x2 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[1] = self.update_attention_scores(attention_scores_per_layer[1], perm, score, node_idx_mapping)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score, node_idx_mapping)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        # Layer 3
        x, (edge_idx3, att_weights3) = self.conv3(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool3(x, edge_index, edge_attr, batch)
        att_weights = self.filter_edges(edge_idx3, att_weights3, perm)

        node_idx_mapping, filenames, edge_idx_mapping, edge_old_idx_new_idx = self.update_node_mapping(node_idx_mapping, perm, original_filenames, edge_index_mapping_rev, edge_index)
        layer_data = self.layer_graph(layer_data, x, edge_index, edge_attr, score, original_node_attr,
                                      att_weights, filenames, node_idx_mapping, edge_index_mapping_rev, edge_old_idx_new_idx, original_edge_attr)
        x3 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[2] = self.update_attention_scores(attention_scores_per_layer[2], perm, score, node_idx_mapping)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score, node_idx_mapping)

        if self.walk_length > 0:
            rwpe = rwpe[perm]
            x = torch.cat([x, rwpe], dim=1)

        # Layer 4
        x, (edge_idx4, att_weights4) = self.conv4(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool4(x, edge_index, edge_attr, batch)
        att_weights = self.filter_edges(edge_idx4, att_weights4, perm)

        node_idx_mapping, filenames, edge_idx_mapping, edge_old_idx_new_idx = self.update_node_mapping(node_idx_mapping, perm, original_filenames, edge_index_mapping_rev, edge_index)
        layer_data = self.layer_graph(layer_data, x, edge_index, edge_attr, score, original_node_attr,
                                      att_weights, filenames, node_idx_mapping, edge_index_mapping_rev, edge_old_idx_new_idx, original_edge_attr)
        x4 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
        attention_scores_per_layer[3] = self.update_attention_scores(attention_scores_per_layer[3], perm, score, node_idx_mapping)
        cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, score, node_idx_mapping)

        normalized_attention_scores_per_layer = [self.normalize_attention_scores(scores) for scores in attention_scores_per_layer]
        normalized_cumulative_scores = self.normalize_attention_scores(cumulative_attention_scores)

        all_patches_per_layer = [
            [(filename, score) for filename, score in zip(original_filenames, layer_scores)]
            for layer_scores in normalized_attention_scores_per_layer
        ]
        all_patches_cumulative = list(zip(original_filenames, normalized_cumulative_scores)) # TODO add patch coordinates here for heatmap & attention graph generation

        x = x1 + x2 + x3 + x4

        return x, all_patches_per_layer, all_patches_cumulative, layer_data


    def filter_edges(self, edge_index, edge_attr, perm):
        row, col = edge_index
        mask = (row[None, :] == perm[:, None]).any(0) & (col[None, :] == perm[:, None]).any(0)
        filtered_edge_attr = edge_attr[mask]
        return filtered_edge_attr

    def layer_graph(self, data, x, edge_index, edge_attr, score, original_node_attr, att_weights, filenames,
                    node_mapping, edge_mapping, edge_old_idx_new_idx, original_edge_attr):
        data.append(Data(x=x,
                         edge_index=edge_index,
                         edge_attr=edge_attr,
                         node_att_scores=score,
                         node_attr=original_node_attr,
                         attention_weights=att_weights,
                         node_filenames=filenames,
                         node_mapping=node_mapping,
                         edge_mapping=edge_mapping,
                         edge_idx_mapping= edge_old_idx_new_idx,
                         original_edge_attr= original_edge_attr
                         ))
        return data

    def update_node_mapping(self, node_mapping, perm, original_filenames, edge_index_mapping_rev, edge_index):
        new_node_mapping = {}
        new_filenames = []
        for new_idx, old_idx in enumerate(perm.tolist()):
            if old_idx in node_mapping:
                original_idx = node_mapping[old_idx]
                new_node_mapping[new_idx] = original_idx
                new_filenames.append(original_filenames[original_idx])

        new_edge_index_mapping = {}
        old_idx_to_new_idx = {}
        for i in range(edge_index.size(1)):
            new_edge_name = edge_index.t()[i].tolist()
            source, target = new_edge_name
            original_edge_name = str([new_node_mapping[source], new_node_mapping[target]])
            old_idx = edge_index_mapping_rev[original_edge_name]
            new_edge_index_mapping[str(new_edge_name)] = old_idx
            old_idx_to_new_idx[old_idx] = i

        return new_node_mapping, new_filenames, new_edge_index_mapping, old_idx_to_new_idx

    def normalize_attention_scores(self, scores):
        # Filter out None values
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return scores  # Return the original list if all layer_data are None

        # Perform min-max normalization on valid layer_data
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) if s is not None else 0 for s in scores]
        else:
            normalized_scores = [1 if s is not None else 0 for s in scores]

        return normalized_scores

    def update_attention_scores(self, attention_scores, perm, score, node_mapping):
        new_attention_scores = [None] * len(attention_scores)
        for new_idx, old_idx in enumerate(perm.tolist()):
            if old_idx in node_mapping:
                new_attention_scores[node_mapping[old_idx]] = score[new_idx].item()
        return new_attention_scores

    def update_cumulative_scores(self, cumulative_scores, perm, score, node_mapping):
        new_cumulative_scores = cumulative_scores.copy()
        for new_idx, old_idx in enumerate(perm.tolist()):
            if old_idx in node_mapping:
                if new_cumulative_scores[node_mapping[old_idx]] is None:
                    new_cumulative_scores[node_mapping[old_idx]] = score[new_idx].item()
                else:
                    new_cumulative_scores[node_mapping[old_idx]] += score[new_idx].item()
        return new_cumulative_scores

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
#             patient_graphs = torch.matmul(x, self.attention_weights) # 1* hidden_dim * 2
#             patient_graphs = F.softmax(patient_graphs, dim= -1)
#             x = torch.sum(x * patient_graphs, dim=0).unsqueeze(0) # hidden_dim * 2
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