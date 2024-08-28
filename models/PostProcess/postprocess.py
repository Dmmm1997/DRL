import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .upsample import NearstUpsample,TransConvUpsample


def make_postprocess(opt):
    postprocess = PostProcess(opt)
    return postprocess


class PostProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.postprocess = self.init_postprocess(opt.model["postprocess"])

    def init_postprocess(self, postprocess_opt):
        postprocess = postprocess_opt.pop("upsample_method")
        if postprocess == "NearstUpsample":
            postprocess = NearstUpsample(**postprocess_opt)
        elif postprocess == "TransConvUpsample":
            postprocess = TransConvUpsample(**postprocess_opt)
        else:
            raise NameError("{} not in the postprocess list!!!".format(postprocess))
        return postprocess

    def forward(self, cls,reg):
        return self.postprocess(cls,reg)
