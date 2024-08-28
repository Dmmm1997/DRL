from torch import nn
from .utils import ChannelPool


class ChannelPooling(nn.Module):
    def __init__(self, head_pool, input_dim, **kwargs):
        super(ChannelPooling, self).__init__()
        self.pool = ChannelPool(mode = head_pool, linear_input_dim=input_dim)

    def forward(self, z, x):
        # zb, zc, zh, zw = z.shape
        x = self.pool(x)
        return x, None
