import sys
sys.path.insert(0,"/home/dmmm/VscodeProject/FPI/")
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from losses.utils import create_labels, create_labels_2
from einops import rearrange


class LocSmoothL1Loss(nn.Module):
    def __init__(self, topk=1, weight_rate=1.0):
        super(LocSmoothL1Loss, self).__init__()
        self.topk = topk
        self.weight_rate = weight_rate

    def forward(self, cls_input, center_rate):
        B,C,H,W = cls_input.shape
        cls_input = torch.sigmoid(cls_input)
        cls_part = cls_input.reshape(cls_input.shape[0], -1)
        topk_indices = torch.topk(cls_part, self.topk, dim=1).indices
        cls_topk_mask = torch.zeros_like(cls_part, device="cuda:0", dtype=bool)
        for i in range(topk_indices.shape[0]):
            cls_topk_mask[i][topk_indices[i]] = True
        cls_topk_mask = cls_topk_mask.reshape(cls_input.shape[0],cls_input.shape[2],cls_input.shape[3])
        # pos_mask = cls_topk_mask.reshape(-1)
        meshgrid_map = self.create_meshgrid_map(cls_input.size())
        meshgrid_pos = []
        for i in range(B):
            cur_meshgrid_map = meshgrid_map[i]
            cur_cls_topk_mask = cls_topk_mask[i]
            cur_meshgrid_pos = cur_meshgrid_map[cur_cls_topk_mask,:]
            meshgrid_pos.append(cur_meshgrid_pos)
        
        meshgrid_pos = torch.stack(meshgrid_pos,dim=0)
        assert meshgrid_pos.shape[1] == self.topk

        center_rate_map = torch.stack(center_rate, dim=1).reshape(B,1,2).repeat((1,self.topk,1)).cuda() #(B,2)
        loc_loss = F.smooth_l1_loss(meshgrid_pos, center_rate_map, reduction='mean')
        return loc_loss * self.weight_rate

    def create_meshgrid_map(self, size):
        b, c, h, w = size
        x = np.arange(h)
        y = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
        loc_map = np.expand_dims(np.concatenate((yy, xx), axis=-1), axis=0).repeat(b, axis=0)
        loc_map_normal = loc_map / [h-1, w-1]
        return torch.from_numpy(loc_map_normal).cuda().float()



if __name__ == "__main__":
    # FIXME 仍然存在一定误差，需要修正。下面的输入输出的损失应为0,目前0.0002
    loss = LocSmoothL1Loss()
    center_rate = [torch.tensor([0.5, 0.2, 0.8, 0.2]), torch.tensor([0.5, 0.5, 0.5, 0.2])]
    # cls_input = torch.randn((4, 1, 24, 24)).cuda()
    cls_input = create_labels((4, 1, 24, 24),center_rate,1)
    # reg_input = torch.randn((4, 2, 24, 24)).cuda()
    loc_loss = loss(cls_input, center_rate)
