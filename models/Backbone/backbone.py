import torch.nn as nn
from .convnext import convnext_small, convnext_tiny
import timm
from .pcpvt import pcpvt_small
from ..utils import vector2array
from .pvt import pvt_small, pvt_tiny
from .mixformer import MixFormer
from .cvt import get_cvt_models
import torch
import os
from .pvtv2 import pvt_v2_b2


def make_backbone(opt, img_size=None):
    backbone_model = Backbone(opt, img_size)
    return backbone_model


class Backbone(nn.Module):
    def __init__(self, opt, img_size):
        super().__init__()
        self.opt = opt
        self.output_index = opt.model["backbone"]["output_index"]
        self.backbone_type = self.opt.model["backbone"]["type"]
        self.pretrain = opt.model["backbone"]["pretrain"]
        self.backbone, self.backbone_out_channel = self.init_backbone(img_size)
        pretrain_path = opt.model["backbone"].get("pretrain_path", None)
        if pretrain_path is not None and os.path.isfile(pretrain_path):
            self.load_checkpoints(pretrain_path)

    def init_backbone(self, img_size):
        backbone = self.backbone_type
        pretrain = self.pretrain
        if backbone == "ResNet50":
            backbone_model = timm.create_model(
                'resnet50', pretrained=pretrain, features_only=True)
            backbone_out_channel = [128, 256, 512, 1024, 2048]
        elif backbone == "ViT-S":
            backbone_model = timm.create_model(
                "vit_small_patch16_224", pretrained=pretrain, img_size=img_size)
            backbone_out_channel = [384]
        elif backbone == "ViT-B":
            backbone_model = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrain, img_size=img_size)
            backbone_out_channel = [768]
        elif backbone == "DeiT-S":
            backbone_model = timm.create_model(
                "deit_small_patch16_224", pretrained=pretrain, img_size=img_size)
            backbone_out_channel = [384]
        elif backbone == "PvT-T":
            backbone_model = pvt_tiny(pretrained=pretrain)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone == "PvT-S":
            backbone_model = pvt_small(pretrained=pretrain)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone == "PcPvT-S":
            backbone_model = pcpvt_small(pretrained=pretrain)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone == "PvTv2-b2":
            # TODO Not completely fitted
            backbone_model = pvt_v2_b2(pretrained=pretrain)
            # backbone_model = timm.create_model(
            #     "pvt_v2_b2", pretrained=pretrain, img_size=img_size)
            backbone_out_channel = [64, 128, 320, 512]
        elif backbone == "ConvneXt-T":
            backbone_model = convnext_tiny(pretrained=pretrain)
            backbone_out_channel = [96, 192, 384, 768]
        elif backbone == "ConvneXt-S":
            backbone_model = convnext_small(pretrained=pretrain)
            backbone_out_channel = [96, 192, 384, 768]
        elif backbone == "EfficientNet-B5":
            backbone_model = timm.create_model(
                "tf_efficientnet_b5", features_only=True, pretrained=pretrain)
            backbone_out_channel = [24, 40, 64, 176, 512]
        elif backbone == "MixFormer":
            backbone_model = MixFormer(**self.opt.model["backbone"])
            backbone_out_channel = backbone_model.embed_dim
        elif backbone == "CvT":
            backbone_model, backbone_out_channel = get_cvt_models(**self.opt.model["backbone"])
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model, [backbone_out_channel[ind] for ind in self.output_index]

    def load_checkpoints(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        filter_ckpt = {k: v for k, v in ckpt.items() if "pos_embed" not in k}
        if self.backbone_type in ["MixFormer"]:
            missing_keys, unexpected_keys = self.backbone.backbone.load_state_dict(ckpt, strict=False)
        else:
            missing_keys, unexpected_keys = self.backbone.load_state_dict(filter_ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        

    def forward(self, image, image_extra=None):
        """backbone forward function

        Args:
            image (torch.tensor): when use mixformer, image means the search image
            image_extra (torch.tensor, optional): when use mixformer, image_extra means the template image. Defaults to None.

        Returns:
            torch.tensor: features extract by the backbone,shape as (B,C,H,W)
        """
        if self.backbone_type in ["ViT-S", "ViT-B"]:
            features = self.backbone.forward_features(image)[:, 1:]
            features = [vector2array(features)]
        if self.backbone_type in ["DeiT-S"]:
            features = self.backbone.forward_features(image)[:, 1:]
            features = [vector2array(features)]
        elif self.backbone_type in ["ResNet50", "EfficientNet-B5"]:
            features = self.backbone(image)
            features = [features[ind] for ind in self.output_index]
        elif self.backbone_type in ["PvT-T", "PvT-S", "PcPvT-S", "ConvneXt-S", "ConvneXt-T", "CvT", "PvTv2-b2"]:
            features = self.backbone.forward_features(image)
            features = [features[ind] for ind in self.output_index]
        elif self.backbone_type in ["MixFormer"]:
            temp_features, search_features = self.backbone(image_extra, image)
            temp_features = [temp_features[ind] for ind in self.output_index]
            search_features = [search_features[ind] for ind in self.output_index]
            return temp_features, search_features
        return features
