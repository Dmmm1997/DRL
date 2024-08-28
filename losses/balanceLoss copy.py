from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class BalanceLoss(nn.Module):
    def __init__(self, center_R, neg_weight=1.0):
        super(BalanceLoss, self).__init__()
        self.center_R = center_R
        # if center_R%2==0:
        #     raise ValueError("center_R must be Odd number")
        self.neg_weight = neg_weight

    def forward(self, cls_input, center_rate):
        # 等等分配
        if self.center_R == 2:
            target = self.create_labels_2(cls_input.size(), center_rate)
        else:
            target = self.create_labels(cls_input.size(), center_rate)

        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        ############## set hanning weight  #####################

        center_num = torch.tensor(self.create_hanning_mask(self.center_R)).unsqueeze(0).repeat(int(target.size(0)),1,1) # 创建汉宁窗口 并且
        pos_center = (center_num!=0)
        weight[pos_mask] = center_num[pos_center].to(torch.float).cuda()
        weight[pos_mask] =weight[pos_mask]/int(target.size(0))

        ############## set normal weight  #####################
        # weight[pos_mask] = 1 / pos_num

        weight[neg_mask] = 1 / neg_num * self.neg_weight  # 为了平衡正负样本
        weight /= weight.sum()  # 归一化除以所有数值之和
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_input, target, weight, reduction='sum')
        # cls_loss = self.focal_loss(cls_input,target,weight)
        # cls_loss =self.binary_cross_entropy_with_logits(cls_input,target,weight)
        return cls_loss

    def focal_loss(self, cls_input, target, weight, alpha=4):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights * weight
        return (pos_loss + neg_loss).sum()

    def create_hanning_mask(self, center_R):
        hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
            np.hanning(center_R + 2),
            np.hanning(center_R + 2))
        hann_window /= hann_window.sum()
        return hann_window[1:-1, 1:-1]

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss + neg_loss).sum()

    def create_labels_2(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.floor(X).reshape(-1, 1)
        intY = np.floor(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            x1, x2, y1, y2 = CenterX, CenterX + self.center_R // 2 + 1, CenterY, CenterY + self.center_R // 2 + 1
            labels[i, 0, int(x1):int(x2), int(y1):int(y2)] = 1
        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch

    def create_labels(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.round(X).reshape(-1, 1)
        intY = np.round(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            pad_right = pad_left = pad_top = pad_bottom = 0
            if CenterX + self.center_R // 2 > h - 1:
                pad_bottom = int(CenterX + self.center_R // 2 - (h - 1))
            if CenterX - self.center_R // 2 < 0:
                pad_top = int(-1 * (CenterX - self.center_R // 2))
            if CenterY + self.center_R // 2 > h - 1:
                pad_right = int(CenterY + self.center_R // 2 - (w - 1))
            if CenterY - self.center_R // 2 < 0:
                pad_left = int(-1 * (CenterY - self.center_R // 2))
            new_label = np.pad(labels[i, 0], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                               constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0] - self.center_R // 2, \
                             new_center[0] + self.center_R // 2 + 1, \
                             new_center[1] - self.center_R // 2, \
                             new_center[1] + self.center_R // 2 + 1  # 为什么＋在创建标签时左闭右开
            label = new_label.copy()
            label[int(x1):int(x2), int(y1):int(y2)] = 1
            label_mask = new_label != -1
            new_label_out = label[label_mask].reshape(h, w)
            labels[i, :] = new_label_out

        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch


class CenterBalanceLoss(nn.Module):
    def __init__(self, center_R, neg_weight=1.0):
        super(CenterBalanceLoss, self).__init__()
        self.center_R = center_R
        self.neg_weight = neg_weight
        # hanning windows
        self.weight_part = self.create_hanning_mask(self.center_R)

    def forward(self, cls_input, center_rate):
        # calc cls loss
        if self.center_R == 2:
            target = self.create_labels_2(cls_input.size(), center_rate)
            weight = self.create_mask_2(cls_input.size(), center_rate)
        else:
            target = self.create_labels(cls_input.size(), center_rate)
            weight = self.create_mask(cls_input.size(), center_rate)
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        neg_num = neg_mask.sum().float()
        weight[pos_mask] = weight[pos_mask] / weight[pos_mask].sum()
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_input, target, weight, reduction='sum')
        return cls_loss

    def create_mask_2(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.floor(X).reshape(-1, 1)
        intY = np.floor(Y).reshape(-1, 1)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        centerxy = np.concatenate((X, Y), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            centerx, centery = centerxy[i]
            x1, x2, y1, y2 = CenterX, CenterX + self.center_R // 2 + 1, CenterY, CenterY + self.center_R // 2 + 1
            weight = np.zeros((self.center_R, self.center_R))
            weight[0][0] = abs(centerx - (CenterX + 1)) * abs(centery - (CenterY + 1))
            weight[0][1] = abs(centerx - CenterX) * abs(centery - (CenterY + 1))
            weight[1][0] = abs(centerx - (CenterX + 1)) * abs((centery - CenterY))
            weight[1][1] = abs(centerx - CenterX) * abs(centery - CenterY)
            labels[i, 0, int(x1):int(x2), int(y1):int(y2)] = weight
        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch

    def create_labels_2(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.floor(X).reshape(-1, 1)
        intY = np.floor(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            x1, x2, y1, y2 = CenterX, CenterX + self.center_R // 2 + 1, CenterY, CenterY + self.center_R // 2 + 1
            labels[i, 0, int(x1):int(x2), int(y1):int(y2)] = 1
        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch

    def create_labels(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.round(X).reshape(-1, 1)
        intY = np.round(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            pad_right = pad_left = pad_top = pad_bottom = 0
            if CenterX + self.center_R // 2 > h - 1:
                pad_bottom = int(CenterX + self.center_R // 2 - (h - 1))
            if CenterX - self.center_R // 2 < 0:
                pad_top = int(-1 * (CenterX - self.center_R // 2))
            if CenterY + self.center_R // 2 > h - 1:
                pad_right = int(CenterY + self.center_R // 2 - (w - 1))
            if CenterY - self.center_R // 2 < 0:
                pad_left = int(-1 * (CenterY - self.center_R // 2))
            new_label = np.pad(labels[i, 0], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                               constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0] - self.center_R // 2, \
                             new_center[0] + self.center_R // 2 + 1, \
                             new_center[1] - self.center_R // 2, \
                             new_center[1] + self.center_R // 2 + 1
            label = new_label.copy()
            label[int(x1):int(x2), int(y1):int(y2)] = 1
            label_mask = new_label != -1
            new_label_out = label[label_mask].reshape(h, w)
            labels[i, :] = new_label_out

        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch

    def create_gaussian_mask(self, radius):

        def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
            """Generate 2D gaussian kernel.

            Args:
                radius (int): Radius of gaussian kernel.
                sigma (int): Sigma of gaussian function. Default: 1.
                dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
                device (str): Device of gaussian tensor. Default: 'cpu'.

            Returns:
                h (Tensor): Gaussian kernel with a
                    ``(2 * radius + 1) * (2 * radius + 1)`` shape.
            """
            x = torch.arange(
                -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
            y = torch.arange(
                -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

            h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

            h[h < torch.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        gaussian_kernel = gaussian2D(radius, sigma=diameter / 6)

        return gaussian_kernel

    def create_hanning_mask(self, center_R):
        hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
            np.hanning(center_R + 2),
            np.hanning(center_R + 2))
        hann_window /= hann_window.sum()
        return hann_window[1:-1, 1:-1]

    def create_mask(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.round(X).reshape(-1, 1)
        intY = np.round(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            pad_right = pad_left = pad_top = pad_bottom = 0
            if CenterX + self.center_R // 2 > h - 1:
                pad_bottom = int(CenterX + self.center_R // 2 - (h - 1))
            if CenterX - self.center_R // 2 < 0:
                pad_top = int(-1 * (CenterX - self.center_R // 2))
            if CenterY + self.center_R // 2 > h - 1:
                pad_right = int(CenterY + self.center_R // 2 - (w - 1))
            if CenterY - self.center_R // 2 < 0:
                pad_left = int(-1 * (CenterY - self.center_R // 2))
            new_label = np.pad(labels[i, 0], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                               constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0] - self.center_R // 2, \
                             new_center[0] + self.center_R // 2 + 1, \
                             new_center[1] - self.center_R // 2, \
                             new_center[1] + self.center_R // 2 + 1
            label = new_label.copy()
            label[int(x1):int(x2), int(y1):int(y2)] = self.weight_part
            label_mask = new_label != -1
            new_label_out = label[label_mask].reshape(h, w)
            labels[i, :] = new_label_out
        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch


class LocLoss(nn.Module):
    def __init__(self):
        super(LocLoss, self).__init__()

    def forward(self, cls_input, loc_input, center_rate):
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
        return loc_loss

    def create_loc_bias(self, size, center_rate):
        b, c, h, w = size
        x = np.arange(h)
        y = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
        loc_map = np.expand_dims(np.concatenate((yy, xx), axis=-1), axis=0).repeat(b, axis=0)
        loc_map_normal = loc_map / [h - 1, w - 1]

        # calculate the bias and generate the bias map
        center_rate = np.stack(center_rate, axis=1)
        center_rate = center_rate[:, np.newaxis, np.newaxis, :]
        # gt = center_rate.repeat(size[0], axis=1).repeat(size[1], axis=2)
        bias_map = center_rate - loc_map_normal
        bias_map *= [h - 1, w - 1]
        return torch.from_numpy(bias_map).cuda().float()


class GaussianFocalLoss(nn.Module):
    def __init__(self, neg_weight=1.0, radius=3):
        super(GaussianFocalLoss, self).__init__()
        self.center_R = 1
        self.neg_weight = neg_weight
        self.radius = radius

    def forward(self, cls_input, center_rate):
        return self.gaussian_loss(cls_input, center_rate, self.radius)

    def gaussian2D(self, radius, sigma=1, dtype=torch.float32, device='cpu'):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int): Sigma of gaussian function. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        x = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

        h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gen_gaussian_target(self, heatmap, center, radius=3, k=1):
        """Generate 2D gaussian heatmap.

        Args:
            heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
                it and maintain the max value.
            center (list[int]): Coord of gaussian kernel's center.
            radius (int): Radius of gaussian kernel.
            k (int): Coefficient of gaussian kernel. Default: 1.

        Returns:
            out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
        """
        diameter = 2 * radius + 1
        gaussian_kernel = self.gaussian2D(
            radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

        x, y = center

        height, width = heatmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                          radius - left:radius + right]
        out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=out_heatmap[y - top:y + bottom, x - left:x + right])

        return out_heatmap

    def gaussian_focal_loss(self, pred, center_rate, radius=3, alpha=2.0, gamma=4.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
        distribution.

        Args:
            pred (torch.Tensor): The prediction.
            gaussian_target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 2.0.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 4.0.
        """
        gaussian_target = torch.zeros_like(pred)
        b, c, h, w = gaussian_target.shape
        for i in range(b):
            center = [int(center_rate[1][i] * (w - 1)), int(center_rate[0][i] * (h - 1))]
            self.gen_gaussian_target(gaussian_target[i, 0], center, radius=radius)

        pred = torch.sigmoid(pred)
        eps = 1e-12
        pos_weights = gaussian_target.eq(1)
        neg_weights = (1 - gaussian_target).pow(gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
        return (pos_loss + neg_loss).sum() / b

    def gaussian_loss(self, cls_input, center_rate, radius):
        # calc cls loss
        target = self.create_labels(cls_input.size(), center_rate)
        # gaussian windows
        gaussian_mask = torch.zeros_like(cls_input)
        b, c, h, w = gaussian_mask.shape
        for i in range(b):
            center = [int(np.round(center_rate[1][i] * (w - 1))), int(np.round(center_rate[0][i] * (h - 1)))]
            self.gen_gaussian_target(gaussian_mask[i, 0], center, radius=radius)
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        # neg_num = neg_mask.sum().float()
        gaussian_mask[pos_mask] = gaussian_mask[pos_mask] / gaussian_mask[pos_mask].sum()
        gaussian_mask[neg_mask] = (1 - gaussian_mask[neg_mask]) / (
                1 - gaussian_mask[neg_mask]).sum() * self.neg_weight
        gaussian_mask /= gaussian_mask.sum()
        # cls_loss = F.binary_cross_entropy_with_logits(
        #     cls_input, target, gaussian_mask, reduction='sum')
        cls_loss = self.focal_loss(cls_input, target, gaussian_mask, alpha=2)
        return cls_loss

    def focal_loss(self, cls_input, target, weight, alpha=2):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights * weight
        return (pos_loss + neg_loss).sum()

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss + neg_loss).sum()

    def create_labels(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = X.int().numpy().reshape(-1, 1)
        intY = Y.int().numpy().reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            pad_right = pad_left = pad_top = pad_bottom = 0
            if CenterX + self.center_R // 2 > h - 1:
                pad_bottom = int(CenterX + self.center_R // 2 - (h - 1))
            if CenterX - self.center_R // 2 < 0:
                pad_top = int(-1 * (CenterX - self.center_R // 2))
            if CenterY + self.center_R // 2 > h - 1:
                pad_right = int(CenterY + self.center_R // 2 - (w - 1))
            if CenterY - self.center_R // 2 < 0:
                pad_left = int(-1 * (CenterY - self.center_R // 2))
            new_label = np.pad(labels[i, 0], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                               constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0] - self.center_R // 2, \
                             new_center[0] + self.center_R // 2 + 1, \
                             new_center[1] - self.center_R // 2, \
                             new_center[1] + self.center_R // 2 + 1
            label = new_label.copy()
            label[int(x1):int(x2), int(y1):int(y2)] = 1
            label_mask = new_label != -1
            new_label_out = label[label_mask].reshape(h, w)
            labels[i, :] = new_label_out

        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch


class LossFunc(nn.Module):
    def __init__(self, center_R, neg_weight=15.0):
        super(LossFunc, self).__init__()
        self.center_R = center_R
        self.neg_weight = neg_weight

        # self.cls_loss = CenterBalanceLoss(center_R=center_R,neg_weight=neg_weight)
        # self.cls_loss = GaussianFocalLoss(neg_weight = self.neg_weight,radius=3)
        self.cls_loss = BalanceLoss(center_R=center_R, neg_weight=self.neg_weight)
        self.loc_loss = LocLoss()
        # self.loc_loss = 0

    def forward(self, input, center_rate):
        cls_input, loc_input = input
        # calc cls loss
        cls_loss = self.cls_loss(cls_input, center_rate)
        # calc loc loss
        if loc_input is not None:
            loc_loss = self.loc_loss(cls_input, loc_input, center_rate)
            # loc_loss = torch.tensor((0))
            return cls_loss, loc_loss
        else:
            return cls_loss, torch.tensor((0))

    def focal_loss(self, cls_input, target, weight, alpha=2):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights * weight
        return (pos_loss + neg_loss).sum()

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss + neg_loss).sum()


if __name__ == '__main__':
    LossFunc(7)(torch.rand((4, 1, 18, 18), device="cuda:0"), np.array([[0.1, 0.4, 0.6, 0.9], [0.9, 0.4, 0.6, 0.9]]))
    # create_loc_bias((2,3,64,64),[[0.034,0.0453],
    #                              [0.4,0.5]])
