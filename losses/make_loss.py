from losses.clsloss import BalanceLoss, CenterBalanceLoss, FocalLoss, CrossEntropyLoss
from losses.regloss import SmoothL1Loss
from losses.locloss import LocSmoothL1Loss
import torch
from torch import nn


def make_loss(opt):
    loss = Loss(opt)
    return loss


class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cls_loss = None
        self.reg_loss = None
        self.loc_loss = None
        # build class loss
        if "cls_loss" in opt.model["loss"]:
            self.cls_loss = self.build_cls_loss(opt.model["loss"]["cls_loss"])
        # build regression loss
        if "reg_loss" in opt.model["loss"]:
            self.reg_loss = self.build_reg_loss(opt.model["loss"]["reg_loss"])
        if "loc_loss" in opt.model["loss"]:
            self.loc_loss = self.build_loc_loss(opt.model["loss"]["loc_loss"])

    def build_cls_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "BalanceLoss":
            loss_func = BalanceLoss(**loss_opt)
        elif loss_type == "CenterBalanceLoss":
            loss_func = CenterBalanceLoss(**loss_opt)
        elif loss_type == "FocalLoss":
            loss_func = FocalLoss(**loss_opt)
        elif loss_type == "CrossEntropyLoss":
            loss_func = CrossEntropyLoss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func

    def build_reg_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "SmoothL1Loss":
            loss_func = SmoothL1Loss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
    
    def build_loc_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "LocSmoothL1Loss":
            loss_func = LocSmoothL1Loss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func

    def forward(self, input, center_rate):
        cls_input, reg_input = input
        # calc cls loss
        if self.cls_loss is not None and cls_input is not None:
            cls_loss = self.cls_loss(cls_input, center_rate)
        # calc reg loss
        if self.reg_loss is not None and reg_input is not None:
            reg_loss = self.reg_loss(cls_input, reg_input, center_rate)
            return cls_loss, reg_loss
        # calc loc loss
        if self.loc_loss is not None:
            loc_loss = self.loc_loss(cls_input, center_rate)
            return cls_loss, loc_loss
        else:
            return cls_loss, torch.tensor((0))
