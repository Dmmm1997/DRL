import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def vector2array(vector):
    n, p, c = vector.shape
    h = w = np.sqrt(p)
    if int(h) * int(w) != int(p):
        raise ValueError("p can not be sqrt")
    else:
        h = int(h)
        w = int(w)
    array = vector.permute(0, 2, 1).contiguous().view(n, c, h, w)
    return array


class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        b,c,h,w = x.shape
        x = x.reshape(b,c,h*w).transpose(-2,-1)
        # x = torch.transpose(x, 1, -1)
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        # x = torch.transpose(x, 1, -1)
        # x = F.avg_pool1d(x, x.size(-1))
        x = x.reshape(b, 1, h,w)
        # x = x.pow(1. / p)
        return x


class ChannelPool(torch.nn.Module):
    """pooling the input shape of (B,C,H,W) in the C dim
    """

    def __init__(self, mode="avg", linear_input_dim=10):
        super(ChannelPool, self).__init__()
        self.mode = mode
        if mode == "conv":
            self.pool = torch.nn.Conv2d(linear_input_dim, 1, kernel_size=1, stride=1, padding=0)
        if mode == "linear":
            self.pool = torch.nn.Linear(linear_input_dim, 1)
        if mode == "gem":
            self.pool = Gem_heat(linear_input_dim)

    def forward(self, x):
        if self.mode == "avg":
            res = torch.mean(x, dim=1, keepdim=True)
        else:
            res = self.pool(x)

        return res


def xcorr_slow(kernel, x):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(kernel, x, padding=True):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    w = kernel.size()[-1]
    pk = kernel.contiguous().view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.contiguous().view(1, -1, x.size()[2], x.size()[3])
    padding_ = w//2 if padding else 0
    po = F.conv2d(px, pk, groups=batch, padding=padding_)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(kernel, x, padding=True):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    w = kernel.size()[-1]
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    padding_ = w//2 if padding else 0
    out = F.conv2d(x, kernel, groups=batch*channel, padding=padding_)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
