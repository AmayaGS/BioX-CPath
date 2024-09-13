import torch
import numpy as np
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, SAGEConv, SAGPooling


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_gflops(model, sample_data):
    total_flops = 0

    def count_conv_flops(layer, x, edge_index, random_walk_pe=None):
        in_features = layer.in_channels
        out_features = layer.out_channels
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if isinstance(layer, GATv2Conv):
            # GAT: matrix mult + attention
            flops = 2 * in_features * out_features * num_edges + 2 * out_features * num_edges
        elif isinstance(layer, GCNConv):
            # GCN: sparse matrix mult
            flops = 2 * in_features * out_features * num_edges
        elif isinstance(layer, SAGEConv):
            # GraphSAGE: dense matrix mult
            flops = 2 * in_features * out_features * num_nodes
        elif isinstance(layer, GINConv):
            # GIN: dense matrix mult in MLP
            mlp = layer.nn
            flops = sum(2 * in_feat * out_feat * num_nodes for in_feat, out_feat in zip(mlp[::2], mlp[1::2]))
        else:
            flops = 0

        if random_walk_pe is not None:
            # Add flops for concatenating random_walk_pe
            flops += num_nodes * random_walk_pe.size(1)

        return flops

    def count_pool_flops(layer, *args):
        # Estimate FLOPs for SAGPooling
        x, edge_index = args[0], args[1]
        num_nodes = x.size(0)
        score_flops = 2 * layer.in_channels * num_nodes  # Score computation
        topk_flops = num_nodes * np.log(num_nodes)  # Approx. for top-k selection
        return score_flops + topk_flops

    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, (GATv2Conv, GCNConv, SAGEConv, GINConv)):
            x, edge_index = input[0], input[1]
            random_walk_pe = x[:, -module.walk_length:] if hasattr(module, 'walk_length') else None
            total_flops += count_conv_flops(module, x, edge_index, random_walk_pe)
        elif isinstance(module, SAGPooling):
            total_flops += count_pool_flops(module, *input)

    hooks = []
    for module in model.modules():
        if isinstance(module, (GATv2Conv, GCNConv, SAGEConv, GINConv, SAGPooling)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        model(sample_data)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops / 1e9  # Convert to GFLOPs


def print_model_stats(model, sample_data):
    num_params = count_parameters(model)
    gflops = estimate_gflops(model, sample_data)

    return num_params, gflops