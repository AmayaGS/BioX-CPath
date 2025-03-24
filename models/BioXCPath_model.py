from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from models.sag_pool_biox import SAGPooling


class LayerConcatSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, dropout_rate):
        super().__init__()
        self.concat_dim = embedding_dim * num_layers
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.multihead_attn = nn.MultiheadAttention(self.concat_dim, num_heads, batch_first=True, dropout=dropout_rate)
        self.pre_norm = nn.LayerNorm(self.concat_dim)
        self.post_norm = nn.LayerNorm(self.concat_dim)

    def forward(self, x):
        # x shape: (1, concat_dim)
        x = x.unsqueeze(1)  # (1, 1, concat_dim)
        x = self.pre_norm(x)
        attn_output = self.multihead_attn(x, x, x, need_weights=False)
        attn_output = self.post_norm(attn_output[0] + x).squeeze(0)
        layer_attention = self.aggregate_layer_attention(attn_output)

        return attn_output, layer_attention

    def aggregate_layer_attention(self, attn_output):
        # attention_weights shape: (1, 1, concat_dim)
        attn_output = attn_output.squeeze(0).squeeze(0)  # (concat_dim,)
        layer_attention = []
        for i in range(self.num_layers):
            start_idx = i * self.embedding_dim
            end_idx = (i + 1) * self.embedding_dim
            layer_attention.append(attn_output[start_idx:end_idx].sum().item())

        layer_attention = torch.tensor(layer_attention)
        # shift to positive values
        layer_attention = layer_attention - layer_attention.min()
        # add epsilon to avoid division by zero
        epsilon = 1e-8
        layer_attention = layer_attention + epsilon
        # Normalize
        layer_attention = layer_attention / layer_attention.sum()

        return layer_attention


class BioXCPath_Classifier(nn.Module):
    def __init__(self, in_features, edge_attr_dim, node_attr_dim, hidden_dim, num_classes, heads, pooling_ratio,
                 walk_length, conv_type, num_layers, embedding_dim, dropout_rate, use_node_embedding=False, use_edge_embedding=False, use_attention=True):
        super(BioXCPath_Classifier, self).__init__()

        self.use_attention = use_attention

        self.bioxcpath = bioxcpath_pooling(in_features, edge_attr_dim, node_attr_dim, hidden_dim, heads, pooling_ratio,
                                           walk_length, conv_type, num_layers, embedding_dim, use_node_embedding, use_edge_embedding)

        concat_dim = hidden_dim * 2 * num_layers
        self.layer_attention = LayerConcatSelfAttention(hidden_dim * 2, num_layers, num_heads=1, dropout_rate=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier1 = nn.Linear(concat_dim, concat_dim // 2)
        self.classifier2 = nn.Linear(concat_dim // 2, num_classes)

    def forward(self, data, label):
        x = self.bioxcpath(data)  # x shape: (1, num_layers * hidden_dim * 2)

        if self.use_attention:
            # Apply self-attention
            x, layer_attention = self.layer_attention(x)
        else:
            layer_attention = None

        # Apply dropout and ReLU
        x = self.dropout(F.relu(x))

        # Classification
        x = self.classifier1(x)
        logits = self.classifier2(x)
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_prob, layer_attention, label


class bioxcpath_pooling(torch.nn.Module):
    def __init__(self, in_features, edge_attr_dim, node_attr_dim, hidden_dim, heads, pooling_ratio, walk_length,
                 conv_type, num_layers, embedding_dim, use_node_embedding=False, use_edge_embedding=False):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.pooling_ratio = pooling_ratio
        self.walk_length = walk_length
        self.use_node_embedding = use_node_embedding
        self.use_edge_embedding = use_edge_embedding
        self.conv_type = conv_type

        if use_edge_embedding:
            self.edge_embedding = nn.Embedding(edge_attr_dim, embedding_dim)
        if use_node_embedding:
            self.node_embedding = nn.Embedding(node_attr_dim, embedding_dim)

        self.convolutions = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_features + walk_length
            if use_node_embedding:
                in_dim += embedding_dim
            if i > 0:
                in_dim = hidden_dim + walk_length
                if use_node_embedding:
                    in_dim += embedding_dim
            conv = self._create_conv_layer(conv_type, in_dim, embedding_dim if use_edge_embedding else None, hidden_dim, heads)
            pool = SAGPooling(hidden_dim, self.pooling_ratio)
            self.convolutions.append(conv)
            self.pooling_layers.append(pool)

    def _create_conv_layer(self, conv_type, in_dim, edge_dim, out_dim, heads):
        if conv_type == "GAT":
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.use_edge_embedding:
            edge_attr_embedded = self.edge_embedding(edge_attr).squeeze()
        else:
            edge_attr_embedded = None

        if self.use_node_embedding:
            node_attr_embedded = self.node_embedding(data.node_attr.unsqueeze(1)).squeeze()
        else:
            node_attr_embedded = None

        if self.walk_length > 0:
            random_walk_pe = data.random_walk_pe

        # Initialise node idx tracking through pooling layers
        node_idx_mapping = {i: i for i in range(x.size(0))}  # k=node,v=original idx
        layer_embeddings = []
        for i in range(self.num_layers):
            if self.walk_length > 0:
                if self.use_node_embedding:
                    x = torch.cat([x, random_walk_pe, node_attr_embedded], dim=1)
                else:
                    x = torch.cat([x, random_walk_pe], dim=1)
            elif self.use_node_embedding:
                x = torch.cat([x, node_attr_embedded], dim=1)

            if self.conv_type in ["GAT"] and self.use_edge_embedding:
                x = F.relu(self.convolutions[i](x, edge_index, edge_attr=edge_attr_embedded))
            else:
                x = F.relu(self.convolutions[i](x, edge_index))

            x, edge_index, edge_attr_embedded, batch, perm, attention_scores, _ = self.pooling_layers[i](x, edge_index, edge_attr_embedded, batch)

            node_idx_mapping = self.update_node_mapping(node_idx_mapping, perm)

            #node_stain_map = [data.node_attr[node_idx_mapping[i]].item() for i in range(len(attention_scores))]

            # SAAPooling
            # Calculate stain-level weights from attention scores
            stain_scores = defaultdict(list)
            for node, score in enumerate(attention_scores):
                stain = data.node_attr[node_idx_mapping[node]].item()
                stain_scores[stain].append(score)

            ## Total proportion of attention per stain for weight calculation
            total_attention = attention_scores.sum()
            stain_weights = {stain: sum(scores) / total_attention for stain, scores in stain_scores.items()}

            avg_pooled_features = []
            max_pooled_features = []
            for stain, stain_weight in stain_weights.items():
                # get stain mask for each node
                # stain_mask = torch.tensor([node_stain_map[i] == stain for i in range(len(x))],
                #                             device=x.device)
                stain_mask = torch.tensor([data.node_attr[node_idx_mapping[i]].item() == stain
                                           for i, _ in enumerate(x)], device=x.device)
                stain_weighted_features = x[stain_mask]

                # Pool features
                avg_pooled_stain = stain_weighted_features.mean(dim=0)
                max_pooled_stain = stain_weighted_features.max(dim=0).values

                # Apply stain-level weight
                avg_pooled_features.append(avg_pooled_stain * stain_weight)
                max_pooled_features.append(max_pooled_stain * stain_weight)

            mean_stain_pooled = torch.stack(avg_pooled_features).sum(dim=0)
            max_stained_pool = torch.stack(max_pooled_features).sum(dim=0)
            layer_embeddings.append(torch.cat([mean_stain_pooled, max_stained_pool]))

            if self.walk_length > 0:
                random_walk_pe = random_walk_pe[perm]
            if self.use_node_embedding:
                node_attr_embedded = node_attr_embedded[perm]

        x = torch.cat(layer_embeddings, dim=0).unsqueeze(0)

        return x

    def update_node_mapping(self, node_mapping, perm):
        new_node_mapping = {}
        for new_idx, old_idx in enumerate(perm.tolist()):
            original_idx = node_mapping[old_idx]
            new_node_mapping[new_idx] = original_idx

        return new_node_mapping
