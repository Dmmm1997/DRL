import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from .pafpn import PAFPN
from .fpn_mmlab import FPN
from .channel_convert import CCN
from .fpn import FPN_I3, FPN_I4


def make_neck(opt):
    backbone_model = Neck(opt)
    return backbone_model


class Neck(nn.Module):
    def __init__(self, opt):
        super().__init__()
        opt.model["neck"]["input_dims"] = opt.backbone_output_channel
        self.opt = opt
        self.neck = self.init_neck(opt.model["neck"])

    def init_neck(self, opt_neck):
        neck = opt_neck["type"]
        if neck == "FPN_I3":
            # neck_model = FPN(opt.backbone_output_channel[:3],opt.neck_output_channel,len(opt.backbone_output_channel))
            neck_model = FPN_I3(**opt_neck)
        elif neck == "FPN_I4":
            neck_model = FPN_I4(**opt_neck)
        elif neck == "FPN":
            neck_model = FPN(**opt_neck)
        elif neck == "PAFPN":
            neck_model = PAFPN(**opt_neck)
        elif neck == "CCN":
            neck_model = CCN(**opt_neck)
        elif neck == "None":
            neck_model = None
        else:
            raise NameError("{} not in the neck list!!!".format(neck))
        return neck_model

    def forward(self, features):
        if self.neck:
            features = self.neck(features)
        return features
