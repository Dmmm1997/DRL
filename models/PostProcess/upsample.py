import torch.nn as nn
import torch.nn.functional as F


class NearstUpsample(nn.Module):
    def __init__(self, output_size, **kwargs):
        super(NearstUpsample, self).__init__()
        self.output_size = output_size
        
    def forward(self, cls_branch, reg_branch):
        if cls_branch is not None:
            cls_branch = F.interpolate(cls_branch, size=self.output_size, mode='nearest')
        if reg_branch is not None:
            reg_branch = F.interpolate(reg_branch, size=self.output_size, mode='nearest')

        return cls_branch,reg_branch


class TransConvUpsample(nn.Module):
    # TODO: add transConv Demo
    def __init__(self, conv_layers=2, **kwargs):
        super(TransConvUpsample, self).__init__()
        cls_branch_list = []
        for _ in range(conv_layers):
            cls_branch_list.append(nn.ConvTranspose2d(1, 10, kernel_size=4, stride=2, padding=1))
            cls_branch_list.append(nn.Conv2d(10,1,kernel_size=3,stride=1,padding=1))

        reg_branch_list = []
        for _ in range(conv_layers):
            reg_branch_list.append(nn.ConvTranspose2d(1, 10, kernel_size=4, stride=2, padding=1))
            reg_branch_list.append(nn.Conv2d(10,1,kernel_size=3,stride=1,padding=1))
        
        self.cls_upsample = nn.Sequential(*cls_branch_list)
        self.reg_upsample = nn.Sequential(*reg_branch_list)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    def forward(self, cls_branch, reg_branch):
        if cls_branch is not None:
            cls_branch = self.cls_upsample(cls_branch)
        if reg_branch is not None:
            reg_branch = self.cls_upsample(reg_branch)
        return cls_branch,reg_branch
