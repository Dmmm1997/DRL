from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from .utils import create_labels, create_labels_2


class SmoothL1Loss(nn.Module):
    def __init__(self, center_R, weight_rate=1.0, score_thr=0.5):
        super(SmoothL1Loss, self).__init__()
        self.center_R = center_R
        self.weight_rate = weight_rate
        self.score_thr = score_thr

    def forward_score(self, cls_input, loc_input, center_rate):
        cls_part = cls_input.reshape(cls_input.shape[0], -1)
        topk_indices = torch.topk(cls_part, 1, dim=1).indices
        cls_topk_mask = torch.zeros_like(cls_part, device="cuda:0", dtype=bool)
        for i in range(topk_indices.shape[0]):
            cls_topk_mask[i][topk_indices[i]] = True
        # pos_mask = pos_mask.reshape(-1)
        # pos_mask = cls_topk_mask.reshape(-1) & pos_mask
        pos_mask = cls_topk_mask.reshape(-1)
        bias_map = self.create_loc_bias(cls_input.size(), center_rate)
        bias_map = bias_map.reshape(-1, 2)
        bias_pos = bias_map[pos_mask].reshape(-1)
        res_pos = loc_input.permute(0, 2, 3, 1).contiguous().reshape(-1, 2)
        res_pos = res_pos[pos_mask].reshape(-1)
        loc_loss = F.smooth_l1_loss(res_pos, bias_pos, reduction='mean')
        return loc_loss * self.weight_rate

    def forward(self, cls_input, loc_input, center_rate):
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)
        reg_mask = (target == 1).reshape(-1)
        cls_mask = cls_input.sigmoid().reshape(-1) > self.score_thr
        mask = reg_mask & cls_mask
        bias_map = self.create_loc_bias(cls_input.size(), center_rate)
        bias_map = bias_map.reshape(-1, 2)
        pos_bias_map = bias_map[mask]
        res_pos_map = loc_input.permute(0, 2, 3, 1).contiguous().reshape(-1, 2)
        res_pos_map = res_pos_map[mask].reshape(-1, 2)
        if mask.sum()==0:
            return torch.tensor(0).to(cls_input.device)
        loc_loss = F.smooth_l1_loss(res_pos_map, pos_bias_map, reduction='mean')
        return loc_loss * self.weight_rate

    def create_loc_bias(self, size, center_rate):
        b, c, h, w = size
        x = np.arange(h)
        y = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
        loc_map = np.expand_dims(np.concatenate((yy, xx), axis=-1), axis=0).repeat(b, axis=0)
        loc_map_normal = loc_map / [h-1, w-1]

        # calculate the bias and generate the bias map
        center_rate = np.stack(center_rate, axis=1)
        center_rate = center_rate[:, np.newaxis, np.newaxis, :]
        # gt = center_rate.repeat(size[0], axis=1).repeat(size[1], axis=2)
        bias_map = center_rate - loc_map_normal
        bias_map *= [h-1, w-1]
        return torch.from_numpy(bias_map).cuda().float()


if __name__ == "__main__":
    loss = SmoothL1Loss(center_R=5)
    cls_input = torch.randn((4, 1, 24, 24)).cuda()
    reg_input = torch.randn((4, 2, 24, 24)).cuda()
    center_rate = [[0.5, 0.2, 0.8, 0.2], [0.5, 0.5, 0.5, 0.2]]
    loc_loss = loss(cls_input, reg_input, center_rate)
