import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmaxp, global_add_pool as gap
from torch_geometric.nn import SAGPooling


class MUSTANG_Classifier(nn.Module):
    def __init__(self, in_features, edge_attr_dim, node_attr_dim, hidden_dim, num_classes, heads, pooling_ratio,
                 walk_length, conv_type, num_layers, embedding_dim):
        super(MUSTANG_Classifier, self).__init__()

        self.krag = mustang_pooling(in_features, edge_attr_dim, node_attr_dim, hidden_dim, heads, pooling_ratio,
                                    walk_length, conv_type, num_layers, embedding_dim)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data, label):
        x = self.krag(data)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        logits = self.lin3(x)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, label

class mustang_pooling(torch.nn.Module):
    def __init__(self, in_features, edge_attr_dim, node_attr_dim, hidden_dim, heads, pooling_ratio, walk_length,
                 conv_type, num_layers, embedding_dim):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.pooling_ratio = pooling_ratio
        self.walk_length = walk_length

        self.edge_embedding = nn.Embedding(edge_attr_dim, embedding_dim)
        self.node_embedding = nn.Embedding(node_attr_dim, embedding_dim)

        self.convolutions = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_features + walk_length + embedding_dim if i == 0 else hidden_dim + walk_length + embedding_dim
            conv = self._create_conv_layer(conv_type, in_dim, embedding_dim, hidden_dim, heads)
            pool = SAGPooling(hidden_dim, self.pooling_ratio)
            self.convolutions.append(conv)
            self.pooling_layers.append(pool)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        edge_attr_embedded = self.edge_embedding(edge_attr)
        node_attr_embedded = self.node_embedding(data.node_attr.unsqueeze(1))

        if self.walk_length > 0:
            random_walk_pe = data.random_walk_pe

        # if hasattr(data, 'node_attr_dim'):
        #     node_attr = data.node_attr.unsqueeze(1)
        # else:
        #     node_attr = torch.zeros(x.size(0), 0, device=x.device)

        layer_embeddings = []
        for i in range(self.num_layers):
            if self.walk_length > 0:
                x = torch.cat([x, random_walk_pe, node_attr_embedded], dim=1)
            else:
                x = torch.cat([x, node_attr_embedded], dim=1)

            x = F.relu(self.convolutions[i](x, edge_index, edge_attr_embedded))
            x, edge_index, edge_attr_embedded, batch, perm, _ = self.pooling_layers[i](x, edge_index, edge_attr_embedded, batch)
            layer_embeddings.append(torch.cat([gmp(x, batch), gmaxp(x, batch)], dim=1))

            if self.walk_length > 0:
                random_walk_pe = random_walk_pe[perm]
            node_attr_embedded = node_attr_embedded[perm]

        x = sum(layer_embeddings)

        return x

    def _create_conv_layer(self, conv_type, in_dim, edge_dim, out_dim, heads):
        if conv_type == "GAT":
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
        elif conv_type == "GCN":
            return GCNConv(in_dim, out_dim, add_self_loops=False)
        elif conv_type == "GraphSAGE":
            return SAGEConv(in_dim, out_dim)
        elif conv_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim)), edge_dim=edge_dim)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

