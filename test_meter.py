# -*- coding: utf-8 -*-

from __future__ import print_function, division

import json

import time
from torch.nn.functional import sigmoid
import yaml
from models.taskflow import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
import os
from datasets.SiamUAV import SiamUAV_test
from collections import defaultdict
from tool.evaltools import Distance
from mmcv import Config


def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='exp_test.py',
                        type=str, help='training dir path')
    parser.add_argument('--k', default=10, type=int, help='')
    opt = parser.parse_args()

    cfg = Config.fromfile(opt.config)
    for key, value in cfg.items():
        setattr(opt, key, value)

    print(opt.data_config["val_batchsize"])

    opt.savename = "result_filterR-{}.txt".format(opt.test_config["filterR"])
    opt.GPS_output_filename = "GPS_pred_gt_filterR-{}.json".format(
        opt.test_config["filterR"])

    return opt


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def create_model(opt):
    model = make_model(opt)
    state_dict = torch.load(opt.test_config["checkpoint"])
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model

def test_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    uav, satellite, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO = zip(*batch)
    return torch.stack(uav, dim=0), torch.stack(satellite, dim=0),X,Y,UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO


def create_dataset(opt):
    dataset_test = SiamUAV_test(opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=opt.data_config["val_batchsize"],
                                              shuffle=False,
                                              num_workers=opt.test_config["num_worker"],
                                              pin_memory=True,
                                              collate_fn = test_collate_fn)
    return dataloaders


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.data_config["Satellitehw"][0]
    y_rate = (pred_Y - label_Y) / opt.data_config["Satellitehw"][1]
    # take the distance to the 0-1
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)
    result = np.exp(-1 * opt.k * distance)
    return result


def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def SDM_evaluateSingle(distance, K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*5e3)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def SDM_evaluate_score(
        opt, UAV_GPS, Satellite_INFO, UAV_image_path, Satellite_image_path, S_X, S_Y):
    # drone/groundtruth GPS info
    drone_GPS_info = [float(UAV_GPS["E"]), float(UAV_GPS["N"])]
    # Satellite_GPS_info format:[tl_E,tl_N,br_E,br_N]
    Satellite_GPS_info = [
        float(Satellite_INFO["tl_E"]),
        float(Satellite_INFO["tl_N"]),
        float(Satellite_INFO["br_E"]),
        float(Satellite_INFO["br_N"])]
    drone_in_satellite_relative_position = [float(Satellite_INFO["center_distribute_X"]),
                                            float(Satellite_INFO["center_distribute_Y"])]
    mapsize = float(Satellite_INFO["map_size"])
    # pred GPS info
    pred_N = Satellite_GPS_info[1] - S_X * \
        ((Satellite_GPS_info[1] - Satellite_GPS_info[3]) / opt.data_config["Satellitehw"][0])
    pred_E = Satellite_GPS_info[0] + S_Y * \
        ((Satellite_GPS_info[2] - Satellite_GPS_info[0]) / opt.data_config["Satellitehw"][1])
    pred_GPS_info = [pred_E, pred_N]
    # calc euclidean Distance between pred and gt
    distance = euclideanDistance(drone_GPS_info, [pred_GPS_info])
    # json_output pred GPS and groundtruth GPS for save
    GPS_output_dict = {}
    GPS_output_dict["GT_GPS"] = drone_GPS_info
    GPS_output_dict["Pred_GPS"] = pred_GPS_info
    GPS_output_dict["UAV_filename"] = UAV_image_path
    GPS_output_dict["Satellite_filename"] = Satellite_image_path
    GPS_output_dict["mapsize"] = mapsize
    GPS_output_dict["drone_in_satellite_relative_position"] = drone_in_satellite_relative_position
    GPS_output_dict["Satellite_GPS_info"] = Satellite_GPS_info
    GPS_output_list.append(GPS_output_dict)
    SDM_single_score = SDM_evaluateSingle(distance, 1)
    return SDM_single_score


GPS_output_list = []


def test(model, dataloader, opt):
    total_score = 0.0
    total_score_b = 0.0
    flag_bias = 0
    start_time = time.time()
    SDM_scores = 0
    MA_dict = defaultdict(int)
    MA_log_list = [1, 3, 5, 10, 20, 30, 50, 100]
    total_samples = 0
    for uav, satellite, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO in tqdm(
            dataloader):
        total_samples+=len(UAV_image_path)
        z = uav.cuda()
        x = satellite.cuda()
        response, loc_bias = model(z, x)
        if opt.model["loss"]["cls_loss"].get("use_softmax", False):
            response = torch.softmax(response,dim=1)[:,1:]
        else:
            response = torch.sigmoid(response)
        maps = response.squeeze().cpu().detach().numpy()
        # 遍历每一个batch
        for ind, map in enumerate(maps):
            # hanning kernel
            if opt.test_config["filterR"]>1:
                kernel = create_hanning_mask(opt.test_config["filterR"])
                map = cv2.filter2D(map, -1, kernel)

            label_XY = np.array([X[ind], Y[ind]])

            satellite_map = cv2.resize(map, opt.data_config["Satellitehw"])
            id = np.argmax(satellite_map)
            S_X = int(id // opt.data_config["Satellitehw"][0])
            S_Y = int(id % opt.data_config["Satellitehw"][1])
            pred_XY = np.array([S_X, S_Y])

            # 获取预测的经纬度信息
            get_gps_x = S_X / opt.data_config["Satellitehw"][0]
            get_gps_y = S_Y / opt.data_config["Satellitehw"][0]
            path = Satellite_image_path[ind].split("/")
            read_gps = json.load(
                open(
                    os.path.join(
                        Satellite_image_path[ind].split("/Satellite")[0],
                        "GPS_info.json"),
                    'r', encoding="utf-8"))
            tl_E = read_gps["Satellite"][path[-1]]["tl_E"]
            tl_N = read_gps["Satellite"][path[-1]]["tl_N"]
            br_E = read_gps["Satellite"][path[-1]]["br_E"]
            br_N = read_gps["Satellite"][path[-1]]["br_N"]
            UAV_GPS_E = read_gps["UAV"]["E"]
            UAV_GPS_N = read_gps["UAV"]["N"]
            PRE_GPS_E = tl_E + (br_E - tl_E) * get_gps_y  # 经度
            PRE_GPS_N = tl_N - (tl_N - br_N) * get_gps_x  # 纬度

            # 统计MA指标
            meter_distance = Distance(
                UAV_GPS_N, UAV_GPS_E, PRE_GPS_N, PRE_GPS_E)
            for meter in MA_log_list:
                if meter_distance <= meter:
                    MA_dict[meter] += 1

            # calculate SDM1 critron
            SDM_single_score = SDM_evaluate_score(
                opt, UAV_GPS[ind], Satellite_INFO[ind], UAV_image_path, Satellite_image_path, S_X, S_Y)
            # SDM score
            SDM_scores += SDM_single_score

            # RDS score
            single_score = evaluate(opt, pred_XY=pred_XY, label_XY=label_XY)
            total_score += single_score
            if loc_bias is not None:
                flag_bias = 1
                loc = loc_bias.squeeze().cpu().detach().numpy()
                id_map = np.argmax(map)
                S_X_map = int(id_map // map.shape[-1])
                S_Y_map = int(id_map % map.shape[-1])
                pred_XY_map = np.array([S_X_map, S_Y_map])
                pred_XY_b = (pred_XY_map + loc[:, S_X_map, S_Y_map]) * \
                    opt.data_config["Satellitehw"][0] / loc.shape[-1]  # add bias
                pred_XY_b = np.array(pred_XY_b)
                single_score_b = evaluate(
                    opt, pred_XY=pred_XY_b, label_XY=label_XY)
                total_score_b += single_score_b

    # print("pred: " + str(pred_XY) + " label: " +str(label_XY) +" score:{}".format(single_score))

    time_consume = time.time() - start_time
    print("time consume is {}".format(time_consume))

    score = total_score / total_samples
    SDM_score = SDM_scores / total_samples
    print("the final RDS score is {}".format(score))
    print("the final SDM score is {}".format(SDM_score))

    if flag_bias:
        score_b = total_score_b / len(dataloader)
        print("the final score_bias is {}".format(score_b))

    # MA@K
    for log_meter in MA_log_list:
        print("MA@{}m = {:.4f}".format(log_meter,
                                       MA_dict[log_meter]/total_samples), end=" | ")
    print()

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    metric_dir = os.path.join(save_dir, "metric_RDS_SDM")
    os.makedirs(metric_dir, exist_ok=True)
    pred_GPS_dir = os.path.join(save_dir, "pred_GPS")
    os.makedirs(pred_GPS_dir, exist_ok=True)
    # write result to txt
    with open(os.path.join(metric_dir, opt.savename), "w") as F:
        F.write("RDS: {}\n".format(score))
        F.write("SDM: {}\n".format(SDM_score))
        for log_meter in MA_log_list:
            F.write("MA@{}m {}\n".format(log_meter,
                    MA_dict[log_meter]/total_samples))
        F.write("time consume is {}".format(time_consume))
    # write mid-process-result to json
    with open(os.path.join(pred_GPS_dir, opt.GPS_output_filename), "w") as F:
        json.dump(GPS_output_list, F, indent=4, ensure_ascii=False)


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()
