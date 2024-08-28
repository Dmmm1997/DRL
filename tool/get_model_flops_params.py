# -*- coding: utf-8 -*-
import sys
sys.path.append("../../")
import argparse

from tool.utils_server import calc_flops_params
from models.taskflow import make_model
from mmcv import Config


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--uav_size', default=128, type=int, help='height')
parser.add_argument('--satellite_size', default=384, type=int, help='width')

opt = parser.parse_args()

cfg = Config.fromfile(opt.config)
for key, value in cfg.items():
    setattr(opt, key, value)

model = make_model(opt).cuda()
model = model.eval()

# thop计算MACs
macs, params = calc_flops_params(
    model, (1, 3, opt.uav_size, opt.uav_size), (1, 3, opt.satellite_size, opt.satellite_size))
print("model MACs={}, Params={}".format(macs, params))
