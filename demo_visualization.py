# -*- coding: utf-8 -*-

from __future__ import print_function, division
import warnings
from models.taskflow import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
from datasets.SiamUAV import SiamUAV_test
warnings.filterwarnings("ignore")
from mmcv import Config
import random
import os
from tool.evaltools import Distance
import json
import shutil
from torch.utils.data import Sampler



def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='checkpoints/#CenterR_train/MixCvT13_FPN128_outstride1_CE_Up2Ori_Balance_cr1.py/MixCvT13_FPN128_outstride1_CE_Up2Ori_Balance_cr1.py',
                        type=str, help='training dir path')
    parser.add_argument('--checkpoint', default="checkpoints/#CenterR_train/MixCvT13_FPN128_outstride1_CE_Up2Ori_Balance_cr1.py/output/net_best.pth", type=str, help='')
    parser.add_argument('--save_nums', default=10, type=int, help='')
    parser.add_argument('--seed', default=666, type=int, help='')
    
    opt = parser.parse_args()
    save_dir = opt.checkpoint.split("/net_best.pth")[0]
    save_dir= os.path.join(save_dir,"visualize")
    opt.save_dir = save_dir

    cfg = Config.fromfile(opt.config)
    for key, value in cfg.items():
        if key not in opt:
            setattr(opt, key, value)
    return opt

def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)

def create_model(opt):
    model = make_model(opt)
    state_dict = torch.load(opt.checkpoint)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model


def create_dataset(opt):

    class CustomSampler(Sampler):
        def __init__(self, data, opt):
            self.data = data
            self.data_len = len(data)
            self.opt = opt
            
        def __iter__(self):
            indices = []
            np.random.seed(self.opt.seed)
            indices = list(range(self.data_len))
            np.random.shuffle(indices)
            return iter(indices)

        def __len__(self):
            return len(self.data)

    dataset_test = SiamUAV_test(opt)
    sampler = CustomSampler(dataset_test, opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                            #   batch_size=1,
                                              sampler=sampler,
                                            #   shuffle=True,
                                              num_workers=opt.data_config["num_worker"],
                                              pin_memory=True)
    return dataloaders


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X-label_X)/opt.Satellitehw[0]
    y_rate = (pred_Y-label_Y)/opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate)+np.square(y_rate))/2)  # take the distance to the 0-1
    result = np.exp(-1*opt.k*distance)
    return result

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalization_v2(data):
    _range = np.max(data)-0
    data = np.maximum(data,0)
    return data/_range

def gen_heatmap(heatmap, ori_image):
    heatmap = normalization(heatmap)
    heatmap = cv2.resize(heatmap, (ori_image.shape[1], ori_image.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, 2)  # 将热力图应用于原始图像model.py
    superimposed_img = heatmap * 0.5 + ori_image *0.5  # 这里的0.4是热力图强度因子
    return superimposed_img

def gen_pred_XY(satellite_map, size):
    id = np.argmax(satellite_map)
    S_X = int(id//size[0])
    S_Y = int(id % size[1])
    pred_XY = np.array([S_X, S_Y])
    return pred_XY

def visualize(model, dataloader, opt):
    cur_nums_count = 0
    for uav, satellite, X, Y, UAV_path, Satellite_path, UAV_GPS, Satellite_INFO in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response,_ = model(z, x)
        response = torch.sigmoid(response)
        map = response.squeeze().cpu().detach().numpy()
        if opt.test_config["filterR"]>1:
            kernel = create_hanning_mask(opt.test_config["filterR"])
            map = cv2.filter2D(map, -1, kernel)
        # h, w = map.shape
        map = cv2.resize(map, opt.data_config["Satellitehw"])
        # predict XY
        pred_XY = gen_pred_XY(map, size=opt.data_config["Satellitehw"])
        # ground XY
        label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])
        # load the image
        uavImage = cv2.imread(UAV_path[0])
        satelliteImage = cv2.imread(Satellite_path[0])
        uavImage = cv2.resize(uavImage, opt.data_config["Satellitehw"])
        satelliteImage = cv2.resize(satelliteImage, opt.data_config["Satellitehw"])
        # generate heatmap
        heatmap = gen_heatmap(map, satelliteImage)
        
        # 获取预测的经纬度信息
        get_gps_x = pred_XY[0] / opt.data_config["Satellitehw"][0]
        get_gps_y = pred_XY[1] / opt.data_config["Satellitehw"][0]
        read_gps = Satellite_INFO
        tl_E = read_gps["tl_E"]
        tl_N = read_gps["tl_N"]
        br_E = read_gps["br_E"]
        br_N = read_gps["br_N"]
        UAV_GPS_E = UAV_GPS["E"]
        UAV_GPS_N = UAV_GPS["N"]
        PRE_GPS_E = tl_E + (br_E - tl_E) * get_gps_y  # 经度
        PRE_GPS_N = tl_N - (tl_N - br_N) * get_gps_x  # 纬度

        # 统计MA指标
        meter_distance = Distance(
            UAV_GPS_N, UAV_GPS_E, PRE_GPS_N, PRE_GPS_E)


        # draw pred and gt point
        heatmap = cv2.circle(
            heatmap, pred_XY[:: -1].astype(int),
            radius=5, color=(255, 0, 0),
            thickness=3)
        heatmap = cv2.circle(
            heatmap, label_XY[:: -1].astype(int),
            radius=5, color=(0, 255, 0),
            thickness=3)
        # puttext meter distance
        heatmap = cv2.putText(heatmap,"{:.2f}m".format(meter_distance),(0,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
        
        # merged_image = np.hstack((uavImage,satelliteImage,heatmap))
        os.makedirs(opt.save_dir, exist_ok=True)
        res_out_dir = os.path.join(opt.save_dir, "res_out")
        os.makedirs(res_out_dir,exist_ok=True)

        save_name = UAV_path[0].split("/")[-3]+"_"+UAV_path[0].split("/")[-1]
        save_name = os.path.join(res_out_dir, save_name)
        # cv2.imwrite(save_name,merged_image)
        cv2.imwrite(save_name, heatmap)

        # save the original uav satellite images
        uav_ori_dir = os.path.join(opt.save_dir, "uav_ori")
        satellite_ori_dir = os.path.join(opt.save_dir, "satellite_ori")
        os.makedirs(uav_ori_dir,exist_ok=True)
        os.makedirs(satellite_ori_dir,exist_ok=True)
        shutil.copyfile(UAV_path[0], os.path.join(uav_ori_dir, UAV_path[0].split("/")[-3]+".jpg"))
        shutil.copyfile(Satellite_path[0], os.path.join(satellite_ori_dir, Satellite_path[0].split("/")[-3]+".jpg"))

        print(save_name)
        cur_nums_count+=1
        if cur_nums_count>=opt.save_nums:
            return
            


def main():
    opt = get_opt()
    setup_seed(opt.seed)
    model = create_model(opt)
    dataloader = create_dataset(opt)
    visualize(model, dataloader, opt)


if __name__ == '__main__':
    main()
