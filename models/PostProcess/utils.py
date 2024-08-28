import numpy as np
import torch
import torch.nn.functional as F


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

    def forward(self, x):
        if self.mode == "avg":
            res = torch.mean(x, dim=1, keepdim=True)
        elif self.mode == "conv":
            res = self.pool(x)
        elif self.mode == "linear":
            res = self.pool(x)
        else:
            res = x

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
