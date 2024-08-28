# -*- coding: utf-8 -*-
import sys
sys.path.append("./")
import argparse
import torch
from tool.utils_server import calc_flops_params
from models.taskflow import make_model
import time
from mmcv import Config
import os


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config', default='ConvNextT_CCN_SGF_Balance_cr1_nw15.py',
                    type=str, help='save model path')
parser.add_argument('--checkpoint', default='./output/net_best.pth',
                    type=str, help='save model path')
parser.add_argument('--calc_nums', default=1000, type=int, help='width')
opt = parser.parse_args()

cfg = Config.fromfile(opt.config)
for key, value in cfg.items():
    setattr(opt, key, value)

model = make_model(opt).cuda()
model = model.eval()

# thop计算MACs
macs, params = calc_flops_params(
    model, (1, 3, opt.data_config["UAVhw"][0], opt.data_config["UAVhw"][1]), (1, 3, opt.data_config["Satellitehw"][0], opt.data_config["Satellitehw"][1]))
input_size_drone = (1, 3, opt.data_config["UAVhw"][0], opt.data_config["UAVhw"][1])
input_size_satellite = (1, 3, opt.data_config["Satellitehw"][0], opt.data_config["Satellitehw"][1])

inputs_drone = torch.randn(input_size_drone).cuda()
inputs_satellite = torch.randn(input_size_satellite).cuda()

print("model MACs={}, Params={}".format(macs, params))

# 预热
for _ in range(10):
    model(inputs_drone,inputs_satellite)

since = time.time()
for _ in range(opt.calc_nums):
    model(inputs_drone,inputs_satellite)


print("inference_time = {}s".format(time.time()-since))


info_save_dir = "./output/metric_effects/"
print(os.makedirs(info_save_dir,exist_ok=True))

info_save_file = os.path.join(info_save_dir,"info.txt")
with open(info_save_file,"w") as F:
    F.write("model MACs={}, Params={}\n".format(macs, params))
    F.write("inference_time = {:.1f}s/{}samples\n".format(time.time()-since,opt.calc_nums))
