from __future__ import absolute_import, print_function

import os
import numpy as np
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import cv2
from .Augmentation import RandomCrop, RandomRotate90, EdgePadding, RandomResize, RotateAndCrop
from torchvision import transforms
from .Augmentation import RandomErasing


class SiamUAV_test(Dataset):
    def __init__(self, opt):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAV_test, self).__init__()
        self.root_dir = opt.data_config["test_dir"]
        mode = opt.data_config["test_mode"]
        self.opt = opt
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir, mode)
        self.seq = glob.glob(os.path.join(self.root_dir_train, "*"))
        self.list_all_info = self.get_total_info()

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:
            UAV = os.path.join(seq, "UAV/0.JPG")
            Satellite_list = glob.glob(os.path.join(seq, "Satellite/*"))
            with open(os.path.join(seq, "labels.json"), 'r', encoding='utf8') as fp:
                json_context = json.load(fp)
            with open(os.path.join(seq, "GPS_info.json"), "r", encoding='utf8') as fp:
                gps_info_context = json.load(fp)
            for s in Satellite_list:
                single_dict = {}
                single_dict["UAV"] = UAV
                single_dict["UAV_GPS"] = gps_info_context["UAV"]
                single_dict["Satellite"] = s
                name = os.path.basename(s)
                single_dict["position"] = json_context[name]
                single_dict["Satellite_INFO"] = gps_info_context["Satellite"][name]
                list_all_info.append(single_dict)
        return list_all_info

    def get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.data_config["UAVhw"], interpolation=3),
            transforms.ToTensor()
        ]

        transform_satellite_list = [
            transforms.Resize(
                self.opt.data_config["Satellitehw"], interpolation=3),
            transforms.ToTensor()
        ]

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]
        UAV_image = Image.open(UAV_image_path)
        UAV_image = self.transform["UAV"](UAV_image)

        Satellite_image_path = single_info["Satellite"]
        Satellite_image_ = Image.open(Satellite_image_path)
        Satellite_image = self.transform["satellite"](Satellite_image_)
        X, Y = single_info["position"]
        X = int(X/Satellite_image_.height *
                self.opt.data_config["Satellitehw"][0])
        Y = int(Y/Satellite_image_.width *
                self.opt.data_config["Satellitehw"][1])

        UAV_GPS = single_info["UAV_GPS"]
        # tl_E,tl_N,br_E,br_N,center_distribute_X,center_distribute_Y,map_size
        Satellite_INFO = single_info["Satellite_INFO"]

        return [UAV_image, Satellite_image, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO]


class SiamUAV_val(Dataset):
    def __init__(self, opt):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAV_val, self).__init__()
        self.opt = opt
        self.transform = self.get_transformer()
        self.val_dir = opt.data_config["val_dir"]
        self.seq = glob.glob(os.path.join(self.val_dir, "*"))
        self.list_all_info = self.get_total_info()

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:
            UAV = os.path.join(seq, "UAV/0.JPG")
            Satellite_list = glob.glob(os.path.join(seq, "Satellite/*"))
            with open(os.path.join(seq, "labels.json"), 'r', encoding='utf8') as fp:
                json_context = json.load(fp)
            for s in Satellite_list:
                single_dict = {}
                single_dict["UAV"] = UAV
                single_dict["Satellite"] = s
                name = os.path.basename(s)
                single_dict["position"] = json_context[name]
                list_all_info.append(single_dict)
        return list_all_info

    def get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.data_config["UAVhw"], interpolation=3),
            transforms.ToTensor()
        ]

        transform_satellite_list = [
            transforms.Resize(
                self.opt.data_config["Satellitehw"], interpolation=3),
            transforms.ToTensor()
        ]

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]
        UAV_image = Image.open(UAV_image_path)
        UAV_image = self.transform["UAV"](UAV_image)

        Satellite_image_path = single_info["Satellite"]
        Satellite_image_ = Image.open(Satellite_image_path)
        Satellite_image = self.transform["satellite"](Satellite_image_)
        X, Y = single_info["position"]
        X = int(X/Satellite_image_.height *
                self.opt.data_config["Satellitehw"][0])
        Y = int(Y/Satellite_image_.width *
                self.opt.data_config["Satellitehw"][1])
        return [UAV_image, Satellite_image, X, Y, UAV_image_path, Satellite_image_path]


class SiamUAVCenter(Dataset):
    def __init__(self, opt):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAVCenter, self).__init__()
        self.opt = opt
        self.transform = self.get_transformers(opt)
        self.root_dir_train = opt.data_config["train_dir"]
        self.seq = glob.glob(os.path.join(self.root_dir_train, "*"))

    def get_transformers(self, opt):
        transform_uav_list = self.get_transformer(
            opt.pipline_config["train_pipeline"]["UAV"])
        transform_satellite_list = self.get_transformer(
            opt.pipline_config["train_pipeline"]["Satellite"])
        transform_satellite_pre_list = self.get_sat_pre_transformer(
            opt.pipline_config["train_pipeline"]["Satellite"])
        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'Satellite': transforms.Compose(transform_satellite_list),
            'Satellite_Pre': transforms.Compose(transform_satellite_pre_list)
        }

        return data_transforms
    
    def get_sat_pre_transformer(self, view_opt):
        transform_list = []
        if "RandomAffine" in view_opt:
            transform_list.append(transforms.RandomAffine(**view_opt["RandomAffine"]))
        if "RandomRotate90" in view_opt:
            transform_list.append(RandomRotate90(**view_opt["RandomRotate90"]))
        if "RandomCrop" in view_opt:
            transform_list.append(RandomCrop(**view_opt["RandomCrop"]))
        return transform_list

    def get_transformer(self, view_opt):
        transform_list = []
        if "RotateAndCrop" in view_opt:
            transform_list.append(RotateAndCrop(**view_opt["RotateAndCrop"]))

        if "RandomAffine" in view_opt:
            transform_list.append(transforms.RandomAffine(**view_opt["RandomAffine"]))

        if "RandomRotate90" in view_opt:
            transform_list.append(RandomRotate90(**view_opt["RandomRotate90"]))

        if "ColorJitter" in view_opt:
            transform_list.append(
                transforms.ColorJitter(**view_opt["ColorJitter"]))

        if "RandomErasing" in view_opt:
            transform_list.append(RandomErasing(**view_opt["RandomErasing"]))

        if "RandomResize" in view_opt:
            transform_list.append(RandomResize(**view_opt["RandomResize"]))

        if "ToTensor" in view_opt:
            transform_list.append(transforms.ToTensor())

        # if self.opt.padding:
        #     transform_uav_list = [EdgePadding(
        #         self.opt.padding)] + transform_uav_list

        return transform_list

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # load the json context
        UAV_image_path = os.path.join(self.seq[index], "UAV", "0.JPG")
        UAV_image = Image.open(UAV_image_path)
        # UAV_image = self.UAVAugmentation(UAV_image)
        # UAV_image.show()
        UAV_image = self.transform["UAV"](UAV_image)

        Satellite_image_path = np.random.choice(
            glob.glob(os.path.join(self.seq[index], "Satellite", "*.tif")), 1)[0]
        Satellite_image = Image.open(Satellite_image_path)

        Satellite_info = self.transform["Satellite_Pre"](Satellite_image)
        
        Satellite_image, [ratex, ratey] = Satellite_info
        
        Satellite_image = self.transform["Satellite"](Satellite_image)

        return [UAV_image, Satellite_image, ratex, ratey]
