# -*- coding: utf-8 -*-
import torch
use_gpu = torch.cuda.is_available()

import gc
gc.enable()


def heatmap_scores(args, graph_net, test_loader, patient_ID, loss_fn, n_classes):
    graph_net.eval()
    attention_scores= {}
    layer_data_dict = {}

    data = test_loader.dataset[0]
    label = test_loader.dataset[1]
    metadata = test_loader.dataset[2]
    filenames = metadata['filenames']

    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        if use_gpu:
            data, label = data.cuda(), label.cuda()

        logits, Y_prob, layer_attention, stain_attention, entropy_scores, all_patches_per_layer, all_patches_cumulative, all_patches_replaced, layer_data = graph_net(data, filenames)

        attention_scores[patient_ID] = {
            'per_layer': all_patches_per_layer,
            'cumulative': all_patches_cumulative,
            'replaced': all_patches_replaced
        }

        layer_data_dict[patient_ID] = layer_data

        Y_hat = Y_prob.argmax(dim=1).detach().cpu()

        if use_gpu:
            del data, logits, Y_prob
            torch.cuda.empty_cache()

    return label.cpu(), Y_hat, metadata, attention_scores, layer_data_dict, layer_attention, stain_attention, entropy_scores

