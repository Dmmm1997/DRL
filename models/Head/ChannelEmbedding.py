from torch import nn
from torch.nn import functional as F
from .utils import ChannelPool


class ChannelEmbedding(nn.Module):
    def __init__(self, input_ndim, mid_process_channels, **kwargs):
        super(ChannelEmbedding, self).__init__()
        last_ndim = input_ndim
        self.mid_process_nums = len(mid_process_channels)
        for module_index, channel in enumerate(mid_process_channels):
            setattr(self, "module_{}".format(module_index), nn.Linear(last_ndim, channel))
            last_ndim = channel
    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, z, x):
        # zb, zc, zh, zw = z.shape
        xb, xc, xh, xw = x.shape
        x = x.contiguous().view(xb, xc, xh*xw).transpose(1, 2).contiguous()
        for ind in range(self.mid_process_nums):
            x = getattr(self, "module_{}".format(ind))(x)
        x = x.transpose(1, 2).view(xb, 1, xh, xw).contiguous()
        return x, None
