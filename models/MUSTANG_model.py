import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling


class MUSTANG_Classifier(torch.nn.Module):

    """Graph Attention Network for full slide graph - https://github.com/AmayaGS/MUSTANG/blob/main/Graph_model.py"""

    def __init__(self, in_features, heads=2, pooling_ratio=0.7):

        super().__init__()

        self.pooling_ratio = pooling_ratio
        self.heads = heads

        self.gat1 = GATv2Conv(in_features, 512, heads=self.heads, concat=False)
        self.gat2 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat3 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat4 = GATv2Conv(512, 512, heads=self.heads, concat=False)

        self.topk1 = SAGPooling(512, pooling_ratio)
        self.topk2 = SAGPooling(512, pooling_ratio)
        self.topk3 = SAGPooling(512, pooling_ratio)
        self.topk4 = SAGPooling(512, pooling_ratio)

        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out



# class MUSTANG_Classifier(nn.Module):
#     def __init__(self, in_features, hidden_dim, num_classes, heads, pooling_ratio, walk_length, conv_type, num_layers):
#         super(MUSTANG_Classifier, self).__init__()
#
#         self.krag = pooling_network(in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type, num_layers)
#
#         self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
#         self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)
#
#     def forward(self, data, label):
#         x = self.krag(data)
#
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         x = F.relu(x)
#         logits = self.lin3(x)
#         Y_prob = F.softmax(logits, dim=1)
#
#         return logits, Y_prob, label
#
#
# class pooling_network(torch.nn.Module):
#     def __init__(self, in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.heads = heads
#         self.pooling_ratio = pooling_ratio
#         self.walk_length = walk_length
#
#         self.convolutions = nn.ModuleList()
#         self.pooling_layers = nn.ModuleList()
#
#         for i in range(num_layers):
#             in_dim = in_features + walk_length if i == 0 else hidden_dim + walk_length
#             conv = self._create_conv_layer(conv_type, in_dim, hidden_dim, heads)
#             pool = SAGPooling(hidden_dim, self.pooling_ratio)
#             self.convolutions.append(conv)
#             self.pooling_layers.append(pool)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         if self.walk_length > 0:
#             random_walk_pe = data.random_walk_pe
#
#         layer_embeddings = []
#         for i in range(self.num_layers):
#             if self.walk_length > 0:
#                 x = torch.cat([x, random_walk_pe], dim=1)
#             x = F.relu(self.convolutions[i](x, edge_index))
#             x, edge_index, _, batch, index, _ = self.pooling_layers[i](x, edge_index, None, batch)
#             layer_embeddings.append(torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1))
#             if self.walk_length > 0:
#                 random_walk_pe = random_walk_pe[index]
#
#         x = sum(layer_embeddings)
#
#         return x
#
#     def _create_conv_layer(self, conv_type, in_dim, out_dim, heads):
#         if conv_type == "GAT":
#             return GATv2Conv(in_dim, out_dim, heads=heads, concat=False)
#         elif conv_type == "GCN":
#             return GCNConv(in_dim, out_dim)
#         elif conv_type == "GraphSAGE":
#             return SAGEConv(in_dim, out_dim)
#         elif conv_type == "GIN":
#             return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim)))
#         else:
#             raise ValueError(f"Unknown conv_type: {conv_type}")

# class MUSTANG_Classifier(nn.Module):
#
#     def __init__(self, in_features, hidden_dim, num_classes, heads, pooling_ratio, walk_length, conv_type):
#
#         super(MUSTANG_Classifier, self).__init__()
#
#         # self.attention = attention
#
#         self.krag = bioxcpath_pooling(in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type)
#
#         # if self.attention:
#         #     self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
#         #     nn.init.xavier_uniform_(self.attention_weights)
#
#         self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
#         self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)
#
#     def forward(self, data, label):
#
#         x = self.krag(data)
#
#         # if self.attention:
#         #     patient_graphs = torch.matmul(x, self.attention_weights) # 1* hidden_dim * 2
#         #     patient_graphs = F.softmax(patient_graphs, dim= -1)
#         #     x = torch.sum(x * patient_graphs, dim=0).unsqueeze(0) # hidden_dim * 2
#
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.lin2(x)
#         x = F.relu(x)
#         logits = self.lin3(x)
#         Y_prob = F.softmax(logits, dim=1)
#
#         return logits, Y_prob, label
#
#
# class bioxcpath_pooling(torch.nn.Module):
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
#     def forward(self, data):
#
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         if self.walk_length > 0:
#             rwpe = data.random_walk_pe
#             #x = torch.cat([x, rwpe], dim=1) # add for counting parameters and gflops. Otherwise, remove.
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, index, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         if self.walk_length > 0:
#             rwpe = rwpe[index]
#             x = torch.cat([x, rwpe], dim=1)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, index, _= self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         if self.walk_length > 0:
#             rwpe = rwpe[index]
#             x = torch.cat([x, rwpe], dim=1)
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, index, _= self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         if self.walk_length > 0:
#             rwpe = rwpe[index]
#             x = torch.cat([x, rwpe], dim=1)
#         x = self.conv4(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, index, _= self.pool4(x, edge_index, None, batch)
#         x4 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         x = x1 + x2 + x3 + x4
#
#         return x
#
# class pooling_network(torch.nn.Module):
#
#     """"""
#
#     def __init__(self, in_features, hidden_dim, heads, pooling_ratio, walk_length, conv_type):
#
#         super().__init__()
#
#         self.heads = heads
#         self.pooling_ratio = pooling_ratio
#
#         if conv_type == "GAT":
#
#             self.conv1 = GATv2Conv(in_features + self.walk_length, hidden_dim, heads=self.heads, concat=False)
#             self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=self.heads, concat=False)
#             self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=self.heads, concat=False)
#             self.conv4 = GATv2Conv(hidden_dim, hidden_dim, heads=self.heads, concat=False)
#
#         self.pool1 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool2 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool3 = SAGPooling(hidden_dim, self.pooling_ratio)
#         self.pool4 = SAGPooling(hidden_dim, self.pooling_ratio)
#
#
#     def forward(self, data):
#
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, _, _= self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, _, _= self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         x = self.conv4(x, edge_index)
#         x = F.relu(x)
#         x, edge_index, _, batch, _, _= self.pool4(x, edge_index, None, batch)
#         x4 = torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1)
#
#         x = x1 + x2 + x3 + x4
#
#         return x