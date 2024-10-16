import os
import sys
import importlib

import torch
import torch.nn as nn

import timm as timm
from timm.layers import to_2tuple

from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vgg16_bn, VGG16_BN_Weights, convnext_base, ConvNeXt_Base_Weights
from transformers import AutoModel


class VGG_embedding(nn.Module):

    """
    VGG16 embedding network for WSI patches
    """

    def __init__(self, embedding_vector_size):

        super(VGG_embedding, self).__init__()
        embedding_net = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)

        # Freeze training for all layers
        for param in embedding_net.parameters():
            param.require_grad = False

        # Newly created modules have require_grad=True by default
        num_features = embedding_net.classifier[6].in_features
        features = list(embedding_net.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, embedding_vector_size)])
        embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier
        self.vgg_embedding = nn.Sequential(embedding_net)

    def forward(self, x):

        output = self.vgg_embedding(x)
        output = output.view(output.size()[0], -1)
        return output



class convNext(nn.Module):

    def __init__(self, embedding_vector_size):

        super(convNext, self).__init__()
        model = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.require_grad = False

        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, embedding_vector_size)
        self.model = nn.Sequential(model)

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class resnet18_embedding(nn.Module):

    def __init__(self, embedding_vector_size):

        super(resnet18_embedding, self).__init__()

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, embedding_vector_size)
        self.model = nn.Sequential(model)

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class contrastive_resnet18(nn.Module):

    def __init__(self, weight_path, embedding_vector_size):

        super(contrastive_resnet18, self).__init__()

        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weight_path), strict=True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, embedding_vector_size)
        self.model = nn.Sequential(model)

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class resnet50_embedding(nn.Module):

    def __init__(self, embedding_vector_size):

        super(resnet50_embedding, self).__init__()

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, embedding_vector_size)
        self.model = nn.Sequential(model)

    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output


class GigaPath_embedding(nn.Module):
    """
    https://huggingface.co/prov-gigapath/prov-gigapath
    """
    def __init__(self, embedding_vector_size):
        super(GigaPath_embedding, self).__init__()

        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.final_layer = nn.Linear(1536, embedding_vector_size)

    def forward(self, x):

        x = self.model(x)
        output = self.final_layer(x)

        return output

class UNI_embedding(nn.Module):
    """
    https://huggingface.co/MahmoodLab/UNI
    """
    def __init__(self, embedding_vector_size):
        super(UNI_embedding, self).__init__()

        self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    def forward(self, x):
        output = self.model(x)

        return output


class BiOptimus_embedding(nn.Module):
    """
    https://huggingface.co/bioptimus/H-optimus-0
    """
    def __init__(self, embedding_vector_size):
        super(BiOptimus_embedding, self).__init__()

        self.model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        self.final_layer = nn.Linear(1536, embedding_vector_size)

    def forward(self, x):

        x = self.model(x)
        output = self.final_layer(x)

        return output


class Phikon_embedding(nn.Module):
    """
    https://huggingface.co/owkin/phikon-v2
    """
    def __init__(self, embedding_vector_size):
        super(Phikon_embedding, self).__init__()

        self.model = AutoModel.from_pretrained("owkin/phikon-v2")

    def forward(self, x):

        x = self.model(x)
        output = x.last_hidden_state[:, 0, :]

        return output

# CTransPath
class ConvStem(nn.Module):
    """
    https://github.com/Xiyue-Wang/TransPath/
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class CTransPath_embedding(nn.Module):
    """
    https://github.com/Xiyue-Wang/TransPath/
    CTransPath uses an older custom version of timm (5.0.4), which needs to be downloaded manually from the following link:
    https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view
    It needs to be called timm_old and placed in the sites-packages folder.
    To avoid conflicts with the newer version of timm, the custom version of timm is imported as timm_old only for this model.
    """
    def __init__(self, weight_path, embedding_vector_size):
        super(CTransPath_embedding, self).__init__()

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                timm_old = importlib.import_module("timm_old")
            except ImportError:
                print("Failed to import timm_old. Make sure it's installed correctly. Please download the custom timm version 5.0.4 from the following link: "
                      "https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view and add it to your site-packages folder.")
                sys.exit(1)

        self.model = timm_old.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()
        self.model.load_state_dict(torch.load(weight_path)['model'], strict=True)
        self.final_layer = nn.Linear(768, embedding_vector_size)

    def forward(self, x):
        x = self.model(x)
        output = self.final_layer(x)

        return output
