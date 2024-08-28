# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


class CCN(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims,
                 num_layers=1,
                 **kwargs):
        super(CCN, self).__init__()
        self.Conv2dDict = nn.ModuleDict()
        self.num_layers = num_layers
        for ind, in_channel in enumerate(input_dims):
            tmp_channels = in_channel
            Conv2dList = nn.ModuleList()
            for _ in range(num_layers):      
                Conv2dList.append(nn.Conv2d(tmp_channels, output_dims, kernel_size=1))
                tmp_channels = output_dims
            conv_module = nn.Sequential(*Conv2dList)
            self.Conv2dDict["neck_ccn_{}".format(ind)] = conv_module
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output_features = []
        for ind in range(len(x)):
            out_feature = x[ind]
            name = "neck_ccn_{}".format(ind)
            out_feature = self.Conv2dDict[name](out_feature)
            output_features.append(out_feature)
        return output_features
