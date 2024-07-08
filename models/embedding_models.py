# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:08:28 2023

@author: AmayaGS
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vgg16_bn, VGG16_BN_Weights, convnext_base, ConvNeXt_Base_Weights



class VGG_embedding(nn.Module):

    """
    VGG16 embedding network for WSI patches
    """

    def __init__(self, embedding_vector_size):
        super(VGG_embedding, self).__init__()
        embedding_net = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)

        # Newly created modules have require_grad=True by default
        num_features = embedding_net.classifier[6].in_features
        features = list(embedding_net.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, embedding_vector_size)])
        embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier
        self.vgg_embedding = nn.Sequential(embedding_net)

        # Freeze training for all layers
        for param in embedding_net.parameters():
            param.require_grad = False

    def forward(self, x):

        output = self.vgg_embedding(x)
        output = output.view(output.size()[0], -1)
        return output



class convNext(nn.Module):

    def __init__(self, embedding_vector_size)

        super(convNext, self).__init__()
        model = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        features = list(model.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, embedding_vector_size)])
        model.classifier = nn.Sequential(*features)
        feature_extractor = nn.Sequential(*list(model.children())[:-1]) #Remove last layer

        for param in feature_extractor.parameters():
            param.require_grad = False

        self.feature = feature_extractor

    def forward(self, x):

        feature = self.feature(x)
        flatten_feature = feature.reshape(feature.size(0), -1)
        return flatten_feature


class resnet18_embedding(nn.Module):

    def __init__(self):

        super(resnet18_embedding, self).__init__()

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = model

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class contrastive_resnet18(nn.Module):

    def __init__(self, weight_path):

        super(contrastive_resnet18, self).__init__()

        MODEL_PATH = weight_path

        model = resnet18(weights=None)
        model.load_state_dict(torch.load(MODEL_PATH), strict=True)
        self.model = model

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class resnet50_embedding(nn.Module):

    def __init__(self):

        super(resnet50_embedding, self).__init__()

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = model

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output