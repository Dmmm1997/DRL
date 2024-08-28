from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

from losses.regloss import SmoothL1Loss
from .utils import create_labels, create_labels_2


class BalanceLoss(nn.Module):
    def __init__(self, center_R, neg_weight=1.0, weight_rate=1.0, use_softmax=False, loss_type = "crossentropy",alpha=5, gamma=2, **kwargs):
        super(BalanceLoss, self).__init__()
        self.center_R = center_R
        # if center_R%2==0:
        #     raise ValueError("center_R must be Odd number")
        self.neg_weight = neg_weight
        self.weight_rate = weight_rate
        self.use_softmax = use_softmax
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_input, center_rate):
        # 等等分配
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)

        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        
        if self.use_softmax:
            if self.loss_type.lower() == "crossentropy":
                cls_loss = self.cross_entropy_loss(cls_input, target, weight)            
        else:
            if self.loss_type.lower() == "crossentropy":
                cls_loss = F.binary_cross_entropy_with_logits(cls_input, target, weight, reduction='sum')
            elif self.loss_type.lower() == "focalloss":
                cls_loss = self.focal_loss(cls_input, target, weight)
            else:
                raise TypeError("the type of the loss_type is not supportted!!")
        return cls_loss * self.weight_rate

    def focal_loss(self, cls_input, target, weight=1.0):
        ce_loss = F.binary_cross_entropy_with_logits(cls_input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss * weight

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.sum()

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-8
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss+neg_loss).sum()
    
    def cross_entropy_loss(self, cls_input, target, weight):
        eps = 1e-8
        # 计算 softmax 函数的值
        softmax_outputs = torch.softmax(cls_input, dim=1)
        
        # 将 labels 转换为 one-hot 编码
        one_hot_labels = torch.zeros_like(cls_input)
        one_hot_labels.scatter_(1, target.long(), 1)
        
        # 计算交叉熵损失
        loss_map = -torch.sum(one_hot_labels * torch.log(softmax_outputs+eps), dim=1,keepdim=True)
        loss = loss_map * weight
        
        return loss.sum()

class CenterBalanceLoss(nn.Module):
    def __init__(self, center_R, neg_weight=1.0, weight_rate=1.0, loss_type = "crossentropy", alpha=5, gamma=2, **kwargs):
        super(CenterBalanceLoss, self).__init__()
        self.center_R = center_R
        self.neg_weight = neg_weight
        # hanning windows
        self.weight_part = self.create_hanning_mask(self.center_R)
        self.weight_rate = weight_rate
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_input, center_rate):
        # calc cls loss
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
            weight = self.create_mask_2(cls_input.size(), center_rate)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)
            weight = self.create_mask(cls_input.size(), center_rate)
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        neg_num = neg_mask.sum().float()
        weight[pos_mask] = weight[pos_mask] / weight[pos_mask].sum()
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()      

        if self.loss_type.lower() == "crossentropy":
            cls_loss = F.binary_cross_entropy_with_logits(cls_input, target, weight, reduction='sum')
        elif self.loss_type.lower() == "focalloss":
            cls_loss = self.focal_loss(cls_input, target, weight)
        else:
            raise TypeError("the type of the loss_type is not supportted!!")
        
        return cls_loss * self.weight_rate
    
    def focal_loss(self, cls_input, target, weight=1.0):
        ce_loss = F.binary_cross_entropy_with_logits(cls_input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss * weight

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.sum()

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
            x1, x2, y1, y2 = CenterX, CenterX+self.center_R//2+1, CenterY, CenterY+self.center_R//2+1
            weight = np.zeros((self.center_R, self.center_R))
            weight[0][0] = abs(centerx-(CenterX+1))*abs(centery-(CenterY+1))
            weight[0][1] = abs(centerx-CenterX)*abs(centery-(CenterY+1))
            weight[1][0] = abs(centerx-(CenterX+1))*abs((centery-CenterY))
            weight[1][1] = abs(centerx-CenterX)*abs(centery-CenterY)
            labels[i, 0, int(x1):int(x2), int(y1):int(y2)] = weight
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
            np.hanning(center_R+2),
            np.hanning(center_R+2))
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
            new_label = np.pad(
                labels[i, 0],
                ((pad_top, pad_bottom),
                 (pad_left, pad_right)),
                'constant', constant_values=(-1, -1))
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
    

class CrossEntropyLoss(nn.Module):
    def __init__(self, center_R, weight_rate=1.0, use_softmax=False, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.center_R = center_R
        self.weight_rate = weight_rate
        self.use_softmax = use_softmax

    def forward(self, cls_input, center_rate):
        # 等等分配
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)

        if self.use_softmax:
            cls_loss = self.cross_entropy_loss(cls_input, target)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_input, target, reduction='mean')
        # cls_loss = self.focal_loss(cls_input,target,weight)
        # cls_loss = self.binary_cross_entropy_with_logits(cls_input, target, weight)
        return cls_loss * self.weight_rate

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-8
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss+neg_loss).sum()
    
    

class FocalLoss(nn.Module):
    def __init__(self, center_R, weight_rate=1.0, alpha=1, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()
        self.center_R = center_R
        self.weight_rate = weight_rate
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_input, center_rate):
        # 等等分配
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)

        cls_loss = self.focal_loss(cls_input,target,)
        return cls_loss * self.weight_rate

    # def focal_loss(self, cls_input, target, weight=1.0, alpha=2):
    #     pred = torch.sigmoid(cls_input)
    #     eps = 1e-8
    #     pos_weights = target.eq(1)
    #     neg_weights = target.eq(0)
    #     pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights * weight
    #     neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights * weight
    #     return (pos_loss+neg_loss).mean()
    

    def focal_loss(self, cls_input, target):
        ce_loss = F.binary_cross_entropy_with_logits(cls_input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean()



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
            center = [int(np.round(center_rate[1][i] * (w - 1))),
                      int(np.round(center_rate[0][i] * (h - 1)))]
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
        return (pos_loss+neg_loss).sum()

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss+neg_loss).sum()

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
            if CenterX+self.center_R//2 > h-1:
                pad_bottom = int(CenterX+self.center_R//2-(h-1))
            if CenterX-self.center_R//2 < 0:
                pad_top = int(-1*(CenterX-self.center_R//2))
            if CenterY+self.center_R//2 > h-1:
                pad_right = int(CenterY+self.center_R//2-(w-1))
            if CenterY-self.center_R//2 < 0:
                pad_left = int(-1*(CenterY-self.center_R//2))
            new_label = np.pad(
                labels[i, 0],
                ((pad_top, pad_bottom),
                 (pad_left, pad_right)),
                'constant', constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0]-self.center_R//2,\
                new_center[0]+self.center_R//2+1,\
                new_center[1]-self.center_R//2,\
                new_center[1]+self.center_R//2+1
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
        self.loc_loss = SmoothL1Loss()

    def forward(self, input, center_rate):
        cls_input, loc_input = input
        # calc cls loss
        cls_loss = self.cls_loss(cls_input, center_rate)
        # calc loc loss
        if loc_input is not None:
            loc_loss = self.loc_loss(cls_input, loc_input, center_rate)
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
        return (pos_loss+neg_loss).sum()

    def binary_cross_entropy_with_logits(self, cls_input, target, weight):
        pred = torch.sigmoid(cls_input)
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = target.eq(0)
        pos_loss = -(pred + eps).log() * pos_weights * weight
        neg_loss = -(1 - pred + eps).log() * neg_weights * weight
        return (pos_loss+neg_loss).sum()


if __name__ == '__main__':
    LossFunc(7)(torch.rand((4, 1, 18, 18), device="cuda:0"),
                np.array([[0.1, 0.4, 0.6, 0.9], [0.9, 0.4, 0.6, 0.9]]))
    # create_loc_bias((2,3,64,64),[[0.034,0.0453],
    #                              [0.4,0.5]])

