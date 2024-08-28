import numpy as np
from torch import nn

def vector2array(vector):
    """convert vector to array shape

    Args:
        vector (torch.tensor): shape of (batch, patches, channels)

    Returns:
        torch.tensor: shape of (batch, channels, height, width)
    """
    n, p, c = vector.shape
    h = w = np.sqrt(p)
    if int(h) * int(w) != int(p):
        raise ValueError("p can not be sqrt")
    else:
        h = int(h)
        w = int(w)
    array = vector.permute(0, 2, 1).contiguous().view(n, c, h, w)
    return array


def get_part(x, padding_time):
    h, w = x.shape[-2:]
    cx, cy = h // 2, w // 2
    ch, cw = h // (padding_time + 1) // 2, w // (padding_time + 1) // 2
    x1, y1, x2, y2 = cx - ch, cy - cw, cx + ch + 1, cy + cw + 1
    part = x[:, :, int(x1):int(x2), int(y1):int(y2)]
    return part


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)