from __future__ import print_function, division

import json

import time
from torch.nn.functional import sigmoid
import yaml
import warnings
from models.model import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
from thop import profile

def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/home/dmmm/FPI', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--checkpoint', default="net_014.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--filterR', default=3, type=int, help='')
    parser.add_argument('--savename', default="result_filterR3.txt", type=str, help='')
    parser.add_argument('--GPS_output_filename', default="GPS_pred_gt_filterR3.json", type=str, help='')
    opt = parser.parse_args()
    config_path = 'opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    opt.UAVhw = config["UAVhw"]
    opt.Satellitehw = config["Satellitehw"]
    opt.share = config["share"]
    opt.backbone = config["backbone"]
    opt.padding = config["padding"]
    opt.centerR = config["centerR"]
    return opt

def create_model(opt):
    model = make_model(opt)
    state_dict = torch.load(opt.checkpoint)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model

opt = get_opt()

model = create_model(opt)

input1 = torch.randn(1,3,112,112).cuda()
input2 = torch.randn(1,3,400,400).cuda()

start_time = time.time()
for _ in range(1000):
    res = model(input1,input2)
print("time_consume:{:.2f}".format(time.time()-start_time))


# flops,params = profile(model,inputs=(input1,input2))
# print("%s|params=%.2f|GFLOPs=%.2f"%("model",params/(1000**2),flops/(1000**3)))

