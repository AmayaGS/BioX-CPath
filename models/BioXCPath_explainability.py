from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SAGPooling
from torch_geometric.data import Data

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


class BioXCPath_explainable_model(nn.Module):
    def __init__(self, in_features, edge_attr_dim, node_attr_dim, hidden_dim, num_classes, heads, pooling_ratio,
                 walk_length, conv_type, num_layers, embedding_dim, dropout_rate, use_node_embedding=False,
                 use_edge_embedding=False, use_attention=True):
        super(BioXCPath_explainable_model, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_attention = use_attention

        self.mustang = bioxcpath_pooling(in_features, edge_attr_dim, node_attr_dim, hidden_dim, heads, pooling_ratio,
                                         walk_length, conv_type, num_layers, embedding_dim, use_node_embedding,
                                         use_edge_embedding)

        concat_dim = hidden_dim * 2 * num_layers
        self.layer_attention = LayerConcatSelfAttention(hidden_dim * 2, num_layers, num_heads=1,
                                                        dropout_rate=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier1 = nn.Linear(concat_dim, concat_dim // 2)
        self.classifier2 = nn.Linear(concat_dim // 2, num_classes)

    def forward(self, data, filenames):
        data = data.to(self.device)
        x, all_patches_per_layer, all_patches_cumulative, all_patches_replaced, layer_data, stain_attention, entropy_scores = self.mustang(data, filenames)  # x shape: (1, num_layers * hidden_dim * 2)

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

        return logits, Y_prob, layer_attention, stain_attention, entropy_scores, all_patches_per_layer, all_patches_cumulative, all_patches_replaced, layer_data

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, data, filenames):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # initialise edge and node embeddings if used.
        if self.use_edge_embedding:
            edge_attr_embedded = self.edge_embedding(edge_attr).squeeze()
        else:
            edge_attr_embedded = None

        if self.use_node_embedding:
            node_attr_embedded = self.node_embedding(data.node_attr.unsqueeze(1)).squeeze()
        else:
            node_attr_embedded = None

        # extract random walk positional embeddings if used
        if self.walk_length > 0:
            random_walk_pe = data.random_walk_pe

        # Initialize list for attention scores for each layer and a cumulative score
        attention_scores_per_layer = [[None for _ in range(len(filenames))] for _ in range(self.num_layers)]
        cumulative_attention_scores = [None for _ in range(len(filenames))]
        replace_attention_scores = [None for _ in range(len(filenames))]
        original_node_attr = data.node_attr # keep original node attributes
        original_edge_attr = edge_attr.clone() # keep original edge attributes
        original_filenames = [filename[0] for filename in filenames] # keep original filenames
        # Create initial node mapping
        node_idx_mapping = {i: i for i in range(x.size(0))} # k=node,v=original idx
        original_edge_mapping = {i: edge_index.t()[i].tolist() for i in range(edge_index.size(1))} # k=edge,v=original idx
        edge_index_mapping_rev = {str(v): k for k, v in original_edge_mapping.items()}

        layer_data = [] # keep layerwise attention scores here
        layer_embeddings = [] # layerwise feature embeddings here
        stain_attention = [] # keep stain attention scores here
        entropy_per_layer = []

        for i in range (self.num_layers):
            if self.walk_length > 0:
                if self.use_node_embedding:
                    x = torch.cat([x, random_walk_pe, node_attr_embedded], dim=1)
                else:
                    x = torch.cat([x, random_walk_pe], dim=1)
            elif self.use_node_embedding:
                x = torch.cat([x, node_attr_embedded], dim=1)

            # extract edge attention weights from GAT
            if self.conv_type in ["GAT"] and self.use_edge_embedding:
                x, (edge_index_gat, att_weights_gat) = self.convolutions[i](x, edge_index, edge_attr=edge_attr_embedded,
                                                                                   return_attention_weights=True)
                x = F.relu(x)
            else:
                x, (edge_index_gat, att_weights_gat) = self.convolutions[i](x, edge_index, return_attention_weights=True)
                x = F.relu(x)

            x, edge_index, edge_attr, batch, perm, attention_scores = self.pooling_layers[i](x, edge_index, edge_attr, batch)
            att_weights = self.filter_edges(edge_index_gat, att_weights_gat, perm)

            node_idx_mapping, filenames, edge_idx_mapping, edge_old_idx_new_idx = self.update_node_mapping(node_idx_mapping, perm, original_filenames,
                                                                                                           edge_index_mapping_rev, edge_index)
            layer_data = self.layer_graph(layer_data, x, edge_index, edge_attr, attention_scores,
                                          original_node_attr, att_weights, filenames, node_idx_mapping,
                                          edge_index_mapping_rev, edge_old_idx_new_idx, original_edge_attr)

            layer_embeddings, normalized_stain_scores, entropy_scores = self.stain_attention_pooling(x, data, perm, attention_scores,
                                                                                     node_idx_mapping, layer_embeddings)
            attention_scores_per_layer[i] = self.update_attention_scores(attention_scores_per_layer[i], perm, attention_scores,
                                                                         node_idx_mapping)
            cumulative_attention_scores = self.update_cumulative_scores(cumulative_attention_scores, perm, attention_scores,
                                                                        node_idx_mapping)
            replace_attention_scores = self.replace_attention_score(replace_attention_scores, perm, attention_scores, node_idx_mapping)

            stain_attention.append(normalized_stain_scores)
            entropy_per_layer.append(entropy_scores)

            if self.walk_length > 0:
                random_walk_pe = random_walk_pe[perm]
            if self.use_node_embedding:
                node_attr_embedded = node_attr_embedded[perm]

        normalized_attention_scores_per_layer = [self.normalize_attention_scores(scores) for scores in
                                                 attention_scores_per_layer]
        normalized_cumulative_scores = self.normalize_attention_scores(cumulative_attention_scores)
        normalized_attention_scores = self.normalize_attention_scores(replace_attention_scores)

        all_patches_per_layer = [
            [(filename, score) for filename, score in zip(original_filenames, layer_scores)]
            for layer_scores in normalized_attention_scores_per_layer
        ]
        all_patches_cumulative = list(zip(original_filenames,
                                          normalized_cumulative_scores))
        all_patches_replace = list(zip(original_filenames,
                                          normalized_attention_scores))

        x = torch.cat(layer_embeddings, dim=0).unsqueeze(0)

        return x, all_patches_per_layer, all_patches_cumulative, all_patches_replace, layer_data, stain_attention, entropy_per_layer


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

    def update_cumulative_scores(self, cumulative_scores, perm, score, node_mapping): # this is wrong
        new_cumulative_scores = cumulative_scores.copy()
        for new_idx, old_idx in enumerate(perm.tolist()):
            if old_idx in node_mapping:
                if new_cumulative_scores[node_mapping[old_idx]] is None:
                    new_cumulative_scores[node_mapping[old_idx]] = score[new_idx].item()
                else:
                    new_cumulative_scores[node_mapping[old_idx]] += score[new_idx].item()
        return new_cumulative_scores

    def replace_attention_score(self, replace_attention_scores, perm, score, node_mapping): # this is wrong
        new_attention_scores = replace_attention_scores.copy()
        for new_idx, old_idx in enumerate(perm.tolist()):
            if old_idx in node_mapping:
                new_attention_scores[node_mapping[old_idx]] = score[new_idx].item()
        return new_attention_scores

    def stain_attention_pooling(self, x, data, perm, attention_scores, node_idx_mapping, layer_embeddings):

        # Normalize attention scores globally
        attention_scores = attention_scores - attention_scores.min()
        attention_scores = attention_scores + 1e-8
        attention_scores = attention_scores / attention_scores.sum()

        entropy_scores = defaultdict(list)
        global_entropy_measure = -1 * (attention_scores * torch.log(attention_scores)).sum().item()
        entropy_scores['global'].append(global_entropy_measure)

        # Weight node features by their attention scores to capture node importance
        weighted_features = x * attention_scores.unsqueeze(-1)

        # Calculate stain-level weights from attention scores
        stain_scores = defaultdict(list)
        for node, score in enumerate(attention_scores):
            stain = data.node_attr[node_idx_mapping[node]].item()
            stain_scores[stain].append(score)

        stain_weights = {stain: torch.tensor(scores).sum() for stain, scores in stain_scores.items()}

        avg_pooled_features = []
        max_pooled_features = []
        for stain, stain_weight in stain_weights.items():
            # get stain mask for each node
            stain_mask = torch.tensor([data.node_attr[node_idx_mapping[i]].item() == stain
                                       for i, _ in enumerate(x)], device=x.device)
            stain_weighted_features = weighted_features[stain_mask]
            stain_attention = attention_scores[stain_mask]
            stain_entropy_measure = -1 * (stain_attention * torch.log(stain_attention)).sum().item()
            entropy_scores[stain].append(stain_entropy_measure)

            # Pool features
            avg_pooled_stain = stain_weighted_features.mean(dim=0)
            max_pooled_stain = stain_weighted_features.max(dim=0).values

            # Apply stain-level weight
            avg_pooled_features.append(avg_pooled_stain * stain_weight)
            max_pooled_features.append(max_pooled_stain * stain_weight)

        mean_stain_pooled = torch.stack(avg_pooled_features).sum(dim=0)
        max_stained_pool = torch.stack(max_pooled_features).sum(dim=0)
        layer_embeddings.append(torch.cat([mean_stain_pooled, max_stained_pool]))

        return layer_embeddings, stain_weights, entropy_scores
