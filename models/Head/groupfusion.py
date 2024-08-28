import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .utils import ChannelPool, xcorr_fast, xcorr_depthwise, xcorr_slow


class SingleGroupFusionHead(nn.Module):
    def __init__(self, input_ndim = 128, mid_ndim = 128, out_scale=0.001, num_classes=1 ,padding=True, enable_reg=False, scale_learnable=True, extra_conv = True, **kwargs):
        super(SingleGroupFusionHead, self).__init__()
        
        self.input_ndim = input_ndim
        self.mid_ndim = mid_ndim
        self.padding = padding
        self.enable_reg = enable_reg

        self.cls_conv_uav = nn.Conv2d(input_ndim, num_classes*mid_ndim, kernel_size=1)
        self.cls_conv_satellite = nn.Conv2d(input_ndim, mid_ndim, kernel_size=1)
        if enable_reg:
            self.reg_conv_uav = nn.Conv2d(input_ndim, 2*mid_ndim, kernel_size=1)
            self.reg_conv_satellite = nn.Conv2d(input_ndim, mid_ndim,kernel_size=1)
            self.loc_adjust = nn.Conv2d(2, 2, kernel_size=1)
        self._reset_parameters()
        self.out_scale = nn.Parameter(torch.tensor(out_scale), requires_grad=True) if scale_learnable else out_scale

    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, z, x):
        # print(self.out_scale)
        if isinstance(z,list):
            z = z[0]
        z_cls = self.cls_conv_uav(z)
        x_cls = self.cls_conv_satellite(x)
        # z_cls, x_cls = z, x
        cls_r = xcorr_fast(z_cls, x_cls, padding = self.padding) * self.out_scale
        # cls_r = xcorr_fast(z_cls, x_cls, padding = self.padding)
        reg_r = None
        if self.enable_reg:
            z_reg = self.reg_conv_uav(z)
            x_reg = self.reg_conv_satellite(x)
            reg_r = self.loc_adjust(xcorr_fast(z_reg, x_reg, padding = self.padding)) * self.out_scale
        return cls_r, reg_r

    
    
class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, padding=True):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        self.padding = padding

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel, padding=self.padding)
        out = self.head(feature)
        return out


class DepthwiseFusion(nn.Module):
    def __init__(self, input_ndim=128, mid_ndim=128, padding=True, enable_reg=False, **kwargs):
        super(DepthwiseFusion, self).__init__()
        self.cls = DepthwiseXCorr(input_ndim, mid_ndim, 1, padding=padding)
        self.padding = padding
        self.enable_reg = enable_reg
        if enable_reg:
            self.loc = DepthwiseXCorr(input_ndim, mid_ndim, 2, padding=padding)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z_f, x_f):
        z_f = z_f[0]
        cls = self.cls(z_f, x_f)
        loc = None
        if self.enable_reg:
            loc = self.loc(z_f, x_f)
        return cls, loc
    

class MultiGroupFusionHead(nn.Module):
    def __init__(self, scale_learnable = True, out_scale=0.001, fusion_type = "SingleGroupFusionHead" , pool = "avg", muti_level_nums=3, input_ndim=128, mid_ndim=128, num_classes=1,padding=True,enable_reg=False, **kwargs):
        super(MultiGroupFusionHead, self).__init__()
        self.GroupFusionList = nn.ModuleList()
        for _ in range(muti_level_nums):
            if fusion_type=="SingleGroupFusionHead":
                single_head_func = SingleGroupFusionHead(input_ndim=input_ndim, mid_ndim = mid_ndim, out_scale=out_scale, num_classes=num_classes ,padding=padding, enable_reg=enable_reg, scale_learnable=scale_learnable)
            self.GroupFusionList.append(single_head_func)
        self.pool = ChannelPool(mode=pool, linear_input_dim=muti_level_nums)
        # if scale_learnable:
            

    def forward(self, z, x):
        # assert len(z)==len(x), "输入尺度不对应！！！"
        cls_branch = []
        reg_branch = []
        for ind, z_part in enumerate(z):
            # cls_branch.append(xcorr_fast(z_part, x)[0])*self.out_scale
            cls, reg = self.GroupFusionList[ind](z_part, x)
            cls_branch.append(cls)
            reg_branch.append(reg)
        res = torch.concat(cls_branch,dim=1)
        res = self.pool(res)
        return res, None
    

class MultiEnhanceGroupFusionHead(nn.Module):
    def __init__(self, 
                 out_scale=0.001, 
                 pool = "avg", 
                 muti_level_nums=4, 
                 gc_mid_channel=32, 
                 single_output_channel=8,
                 merged_linear_layer_num=1,
                 merged_linear_channel=128,
                 **kwargs):
        super(MultiEnhanceGroupFusionHead, self).__init__()
        self.out_scale = out_scale
        self.gc_mid_channel = gc_mid_channel
        self.pool = ChannelPool(mode=pool, linear_input_dim=muti_level_nums)
        self.merged_linear = nn.Conv2d(muti_level_nums*single_output_channel, merged_linear_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, z, x):
        # assert len(z)==len(x), "输入尺度不对应！！！"
        res = []
        for z_part in z:
            res.append(self._fast_xcorr(z_part, x) * self.out_scale)
        res = torch.concat(res,dim=1)
        res = self.merged_linear(res)
        res = self.pool(res)
        return res

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()

        x = x.contiguous().view(-1, nz * c, h, w).contiguous()
        # enhance
        gc_out_channel = zc//self.gc_mid_channel
        z = z.contiguous().view(zn, gc_out_channel, self.gc_mid_channel, zh, zw).contiguous().view(zn*gc_out_channel, self.gc_mid_channel, zh, zw).contiguous()
        group = gc_out_channel*zn

        out = F.conv2d(x, z, groups=group, padding=zh // 2)

        out = out.view(nx, -1, out.size(-2), out.size(-1)).contiguous()
        return out
    