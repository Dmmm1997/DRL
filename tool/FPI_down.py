# -*- coding: utf-8 -*-

from __future__ import print_function, division

import glob

import yaml
import warnings
from models.model import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
from tool.get_property import find_GPS_image
from datasets.SiamUAV import SiamUAV_test
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

uav_root=r"E:\FPI\Deit_S_olddata\A_MAP\uav_down\jiliang"
sat_root=r"E:\FPI\Deit_S_olddata\A_MAP\maps\jiliang.tif"
cut_sa_size_h = [1000, 5000]  # 调整显示的卫星图高度
cut_sa_size_w = [4000, 7800]  # 调整显示的卫星图宽度

def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--checkpoint', default="../net_016.pth", type=str, help='save_model')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--cuda', default=3, type=int, help='use panet or not')
    parser.add_argument('--cut_size', default=2400, type=int, help='卫星图裁切的大小')

    opt = parser.parse_args()
    config_path = '../opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    opt.UAVhw = config["UAVhw"]
    opt.Satellitehw = config["Satellitehw"]
    opt.share = config["share"]
    opt.backbone = config["backbone"]
    opt.padding = config["padding"]
    opt.centerR = config["centerR"]
    return opt


def create_model(opt):
    # torch.cuda.set_device(opt.cuda)
    model = make_model(opt)
    state_dict = torch.load(opt.checkpoint)
    pretrained_dict = {k.split("module.")[-1]: v for k, v in state_dict.items()}
    model.load_state_dict(pretrained_dict)
    # model = model.cuda()
    model.eval()
    return model


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.Satellitehw[0]
    y_rate = (pred_Y - label_Y) / opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * opt.k * distance)
    return result


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R + 2),
        np.hanning(center_R + 2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def test(z, x, model, opt, X, Y, uav_pic,roi):
    # z = uav.cuda()
    # x = satellite.cuda()
    response, _ = model(z, x)
    map1 = response.squeeze().cpu().detach().numpy()
    kernel = create_hanning_mask(opt.centerR)  # 21
    map1 = cv2.filter2D(map1, -1, kernel)
    # h, w = map.shape
    satellite_map1 = cv2.resize(map1, opt.Satellitehw)

    id1 = np.argmax(satellite_map1)

    S_X1 = int(id1 // opt.Satellitehw[0])
    S_Y1 = int(id1 % opt.Satellitehw[1])
    point_x = (float(S_X1) / opt.Satellitehw[0] * opt.cut_size) - int(opt.cut_size / 2)  # 在实际卫星图片中预测位置移动的距离
    point_y = (float(S_Y1) / opt.Satellitehw[0] * opt.cut_size) - int(opt.cut_size / 2)
    X = X + point_y  # 预测出来的位置X
    Y = Y + point_x  # 预测出来的位置y

    pred_XY1 = np.array([S_X1, S_Y1])

    satelliteImage = cv2.resize(roi, opt.Satellitehw)

    result_picture = cv2.circle(satelliteImage, pred_XY1[::-1].astype(int), radius=5, color=(255, 0, 0), thickness=3)

    return int(X), int(Y), result_picture


def getPosInfo(imgPath):
    GPS_info = find_GPS_image(imgPath)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    E = y[3]
    N = y[1]
    return [N, E]


def get_transformer(opt):
    transform_uav_list = [
        transforms.Resize(opt.UAVhw, interpolation=3),
        transforms.ToTensor()
    ]

    transform_satellite_list = [
        transforms.Resize(opt.Satellitehw, interpolation=3),
        transforms.ToTensor()
    ]

    data_transforms = {
        'UAV': transforms.Compose(transform_uav_list),
        'satellite': transforms.Compose(transform_satellite_list)
    }

    return data_transforms


def crop_image(im):
    width, height = im.size
    if width < height:
        new_height = new_width = width
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        crop_im = im.crop((left, top, right, bottom))  # Cropping Image
    else:
        new_height = new_width = height
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        crop_im = im.crop((left, top, right, bottom))  # Cropping Image

    # crop_im.save(file_name+"_new.jpg")  #Saving Images
    return crop_im


if __name__ == '__main__':
    opt = get_opt()
    pic_transform = get_transformer(opt)
    AllImage = cv2.imread(sat_root)  # 读取卫星图片
    AllImage_show = AllImage.copy()  # 用于绘制的卫星图片
    h, w, c = AllImage.shape
    print("sat_h:{}".format(h))
    print("sat_w:{}".format(w))

    if (cut_sa_size_w[1] > w or cut_sa_size_h[1] > h):
        raise ValueError("图片裁切尺度超出卫星图大小")

    AllImage_show = AllImage_show[cut_sa_size_h[0]:cut_sa_size_h[1], cut_sa_size_w[0]:cut_sa_size_w[1],
                    :]  # 先高y 再宽x cv2.resize 先x宽 再y高

    with open((sat_root.split(sat_root.split("\\")[-1])[0]+"pos.txt"), 'r', encoding='UTF-8') as F:  # 获取卫星图片的经纬度信息
        listLine = F.readlines()
        for line in listLine:
            name, TN, TE, BN, BE = line.split(" ")
            if name == uav_root.split("\\")[-1]:
                startE = eval(TE.split("TE")[-1])
                startN = eval(TN.split("TN")[-1])
                endE = eval(BE.split("BE")[-1])
                endN = eval(BN.split("BN")[-1])

    model = create_model(opt)
    first_imgpath = glob.glob(uav_root+"/*")[0]
    img = Image.open(first_imgpath)  # 读取第一帧无人机图片
    queryPosInfo = getPosInfo(first_imgpath)  # 获取第一帧无人机图片的经纬度信息
    Q_N = float(queryPosInfo[0])
    Q_E = float(queryPosInfo[1])
    X = int((Q_E - startE) / (endE - startE) * w)  # 无人机在卫星图片中的实际位置w
    Y = int((Q_N - startN) / (endN - startN) * h)

    for num, uav_pic in enumerate(glob.glob(uav_root+"/*")):
        ################获取无人机图##################################
        uavin = Image.open(uav_pic)  # 打开无人机图片
        uavin = crop_image(uavin)  # 中心裁切无人机图片
        uav_show = np.array(uavin)
        uavin = pic_transform["UAV"](uavin)
        uavin = torch.unsqueeze(uavin, 0)  # 扩充维度便于放入模型推理

        ###########################################################

        ################获取卫星图######################################
        roi_bbox = [Y - int(opt.cut_size / 2), Y + int(opt.cut_size / 2), X - int(opt.cut_size / 2),
                    X + int(opt.cut_size / 2)]  # 上下左右
        roi_bbox_top=roi_bbox_bottom=roi_bbox_left=roi_bbox_right=0
        if roi_bbox[0] < 0:
            roi_bbox_top = int(-1 * roi_bbox[0])
            roi_bbox[0]=0
        if roi_bbox[1] > h:
            roi_bbox_bottom = int(roi_bbox[1]-h)
            roi_bbox[1] = h
        if roi_bbox[2] < 0:
            roi_bbox_left = int(-1 * roi_bbox[2])
            roi_bbox[2] = 0
        if roi_bbox[3] > w:
            roi_bbox_right = int(roi_bbox[3]-w)
            roi_bbox[3] = w


        roi = AllImage[int(roi_bbox[0]):int(roi_bbox[1]), int(roi_bbox[2]):int(roi_bbox[3]), :]
        mean = np.mean(roi, axis=(0, 1), dtype=float)
        # padding the border
        roi = cv2.copyMakeBorder(roi, roi_bbox_top, roi_bbox_bottom, roi_bbox_left, roi_bbox_right,
                                          cv2.BORDER_CONSTANT,
                                          value=mean)

        sain = pic_transform["satellite"](Image.fromarray(roi))  # 打开裁切好的卫星tupian
        sain = torch.unsqueeze(sain, 0)
        ###########################################################

        ########################  匹配  ###################################
        X, Y, result_picture = test(uavin, sain, model, opt, X, Y,
                                    uav_pic,roi)  # 输入的 X Y 是当前模型预测的位置 第一次输入为第一帧无人机图片的经纬度信息所给的位置
        ###########################################################

        queryPosInfo = getPosInfo(uav_pic)
        Q_N = float(queryPosInfo[0])
        Q_E = float(queryPosInfo[1])
        X_REAL = int((Q_E - startE) / (endE - startE) * w)
        Y_REAL = int((Q_N - startN) / (endN - startN) * h)

        if num >= 10:
            cv2.circle(AllImage_show, (X_REAL - cut_sa_size_w[0], Y_REAL - cut_sa_size_h[0]), 40, color=(0, 0, 255),
                       thickness=6)  # 红色真实点
            cv2.putText(AllImage_show, str(num), (X_REAL - 40 - cut_sa_size_w[0], Y_REAL + 20 - cut_sa_size_h[0]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (0, 0, 255), 3, cv2.LINE_AA)

            cv2.circle(AllImage_show, (X - cut_sa_size_w[0], Y - cut_sa_size_h[0]), 40, color=(0, 255, 0), thickness=6)
            cv2.putText(AllImage_show, str(num), (X - 40 - cut_sa_size_w[0], Y + 20 - cut_sa_size_h[0]),
                        cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.circle(AllImage_show, (X_REAL - cut_sa_size_w[0], Y_REAL - cut_sa_size_h[0]), 40, color=(0, 0, 255),
                       thickness=6)  # 红色真实点
            cv2.putText(AllImage_show, str(num), (X_REAL - 25 - cut_sa_size_w[0], Y_REAL + 20 - cut_sa_size_h[0]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (0, 0, 255), 3, cv2.LINE_AA)

            cv2.circle(AllImage_show, (X - cut_sa_size_w[0], Y - cut_sa_size_h[0]), 40, color=(0, 255, 0), thickness=6)
            cv2.putText(AllImage_show, str(num), (X - 25 - cut_sa_size_w[0], Y + 20 - cut_sa_size_h[0]),
                        cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 3, cv2.LINE_AA)

        # cv2.circle(AllImage_show, (X_REAL, Y_REAL), 30, color=(0, 0, 255), thickness=10) # 红色真实点
        # cv2.circle(AllImage_show, (X, Y), 30, color=(0, 255, 0), thickness=10)
        # cv2.imwrite(r"E:\pycharm_project\mixformer_down/top_{}.tif".format(University), AllImage)

        uav_show = cv2.imread(uav_pic)
        cv2.imshow("uav_match", cv2.resize(uav_show, [400, 400]))  # 显示需要匹配的无人机图

        cv2.imshow("result_picture", cv2.resize(result_picture, [400, 400]))  # 显示需要匹配的无人机图

        cv2.imshow("all", cv2.resize(AllImage_show, [2 * int((cut_sa_size_w[1] - cut_sa_size_w[0]) / 10),
                                                     2 * int((cut_sa_size_h[1] - cut_sa_size_h[0]) / 10)]))
        cv2.waitKey(0)