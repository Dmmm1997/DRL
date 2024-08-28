import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from .AttentionFusionLib import AttentionFusionLib
from .attentionfusion import CrossAttentionFusion, MultiCrossAttentionFusion, AttentionFusionBlock
from .groupfusion import MultiGroupFusionHead, SingleGroupFusionHead, MultiEnhanceGroupFusionHead, DepthwiseFusion
from .ChannelEmbedding import ChannelEmbedding
from .ChannelPooling import ChannelPooling


def make_head(opt):
    head_model = Head(opt)
    return head_model


class Head(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.head = self.init_head(opt.model["head"])

    def init_head(self, head_opt):
        head = head_opt.pop("type")
        if head == "SingleGroupFusionHead":
            head_model = SingleGroupFusionHead(**head_opt)
        elif head == "MultiGroupFusionHead":
            head_model = MultiGroupFusionHead(**head_opt)
        elif head == "CrossAttentionFusion":
            head_model = CrossAttentionFusion(self.opt)
        elif head == "MultiCrossAttentionFusion":
            head_model = MultiCrossAttentionFusion(self.opt)
        elif head == "MultiEnhanceGroupFusionHead":
            head_model = MultiEnhanceGroupFusionHead(**head_opt)
        elif head == "AttentionFusionLib":
            head_model = AttentionFusionLib(**head_opt)
        elif head == "ChannelEmbedding":
            head_model = ChannelEmbedding(**head_opt)
        elif head == "AttentionFusionBlock":
            head_model = AttentionFusionBlock(self.opt)
        elif head == "DepthwiseFusion":
            head_model = DepthwiseFusion(**head_opt)
        elif head == "ChannelPooling":
            head_model = ChannelPooling(**head_opt)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, z, x):
        return self.head(z, x)
