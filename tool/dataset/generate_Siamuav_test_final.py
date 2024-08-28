import cv2
import numpy as np
import os
import glob

import pylab as p
from tqdm import tqdm
from get_property import find_GPS_image
import json
from multiprocessing import Pool
from PIL import Image
import math

SATELLITE_NUM_Repeat = 1
map_range = [700,1800]
stride = 100
# GPS_info_dict = {"UAV":{},"Satellite":{}}
index = 0
Satellite_output_size = 768

class RandomCrop(object):
    """
    random crop from satellite and return the changed label
    """
    def __init__(self, cover_rate=0.9,map_size=(500,1200)):
        """
        map_size: (low_size,high_size)
        cover_rate: the max cover rate
        """
        self.cover_rate = cover_rate
        self.map_size = map_size


    def __call__(self, img, mapsize = None):
        if not mapsize:
            map_size = round(np.random.randint(int(self.map_size[0]),int(self.map_size[1]))/stride)*stride
        else:
            map_size = mapsize
        if map_size%2==1:
            map_size-=1
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h,w,c = img.shape
        cy,cx = h//2,w//2
        # bbox = np.array([cx-self.map_size//2,cy-self.map_size//2,cx+self.map_size//2,cy+self.map_size//2],dtype=np.int)
        # new_map = img[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1,:]
        # assert new_map.shape[0:2] == [self.map_size,self.map_size], "the size is not correct"
        RandomCenterX = np.random.randint(int(0.5*w-self.cover_rate/2*map_size),int(0.5*w+self.cover_rate/2*map_size))
        RandomCenterY = np.random.randint(int(0.5*h-self.cover_rate/2*map_size),int(0.5*h+self.cover_rate/2*map_size))
        center_distribute_X = (RandomCenterX-0.5*w)/map_size*2
        center_distribute_Y = (RandomCenterY-0.5*h)/map_size*2
        bbox = np.array([RandomCenterX-map_size//2,
                         RandomCenterY-map_size//2,
                         RandomCenterX+map_size//2,
                         RandomCenterY+map_size//2],dtype=int)
        res_bbox = bbox.copy()
        bias_left = bias_top = bias_right = bias_bottom = 0
        if bbox[0] < 0:
            bias_top = -int(bbox[0])
            bbox[0] = 0
        if bbox[1] < 0:
            bias_left = -int(bbox[1])
            bbox[1] = 0
        if bbox[2] >= w:
            bias_bottom = int(bbox[2] - w)
            bbox[2] = w
        if bbox[3] >= h:
            bias_right = int(bbox[3] - h)
            bbox[3] = h
        croped_image = img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3]), :]
        
        mean = np.mean(croped_image, axis=(0, 1), dtype=float)
        # padding the border
        croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right,
                                          cv2.BORDER_CONSTANT,
                                          value=mean)
        ratex = 0.5+(cx-RandomCenterX)/map_size
        ratey = 0.5+(cy-RandomCenterY)/map_size
        # cv2.imshow("sf",cv2.circle(croped_image,(int(ratey*map_size),int(ratex*map_size)),radius=5,color=(255,0,0),thickness=2))
        # cv2.waitKey(0)

        assert croped_image.shape[0:2] == (map_size, map_size), "the size is not correct the cropped size is {},the map_size is {}".format(croped_image.shape[:2],map_size)
        # image = Image.fromarray(croped_image.astype('uint8')).convert('RGB')
        X = int(ratex*map_size)
        Y = int(ratey*map_size)
        return croped_image,[X,Y],res_bbox,[center_distribute_X,center_distribute_Y],map_size


# from center crop image
def center_crop_and_resize(img,target_size=None):
    h,w,c = img.shape
    min_edge = min((h,w))
    if min_edge==h:
        edge_lenth = int((w-min_edge)/2)
        new_image = img[:,edge_lenth:w-edge_lenth,:]
    else:
        edge_lenth = int((h - min_edge) / 2)
        new_image = img[edge_lenth:h-edge_lenth, :, :]
    assert new_image.shape[0]==new_image.shape[1],"the shape is not correct"
    # # LINEAR Interpolation
    if target_size:
        new_image = cv2.resize(new_image,target_size)

    return new_image


def resize(img,target_size):
    return cv2.resize(img,target_size)

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number

# get position(E,N) from the txt file
def getMapPosition(txt):
    place_info_dict = {}
    with open(txt) as F:
        context = F.readlines()
        for line in context:
            name = line.split(" ")[0]
            TN = float(line.split((" "))[1].split("TN")[-1])
            TE = float(line.split((" "))[2].split("TE")[-1])
            BN = float(line.split((" "))[3].split("BN")[-1])
            BE = float(line.split((" "))[4].split("BE")[-1])
            place_info_dict[name] = [TN, TE, BN, BE]
    return place_info_dict

# check and makedir
def checkAndMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def CenterCropFromSatellite(Mapimage,Position,mapsize,target_size=None):
    centerX,centerY = Position
    new_X_min = centerX-mapsize//2
    new_X_max = centerX+mapsize//2
    new_Y_min = centerY-mapsize//2
    new_Y_max = centerY+mapsize//2
    h, w, c = Mapimage.shape
    bbox = [new_X_min, new_Y_min, new_X_max, new_Y_max]
    bias_left = bias_top = bias_right = bias_bottom = 0
    if bbox[0] < 0:
        bias_left = -math.floor(bbox[0])
        bbox[0] = 0
    if bbox[1] < 0:
        bias_top = -math.floor(bbox[1])
        bbox[1] = 0
    if bbox[2] >= w:
        bias_right = math.floor(bbox[2] - w)
        bbox[2] = w
    if bbox[3] >= h:
        bias_bottom = math.floor(bbox[3] - h)
        bbox[3] = h
    croped_image = Mapimage[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    mean = np.mean(croped_image, axis=(0, 1), dtype=float)
    # padding the border
    croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT,
                                      value=mean)
    if target_size is not None:
        croped_image = cv2.resize(croped_image,target_size)
    return croped_image


def pool_func(JPG):
    # image
    image = cv2.imread(JPG)
    # position info
    GPS_info = find_GPS_image(JPG)
    y = list(list(GPS_info.values())[0].values())
    E, N = y[3], y[1]
    # # compute the corresponding position of the big satellite image
    # centerX_2019 = (E - cur_TE_2019) / (cur_BE_2019 - cur_TE_2019) * map_w_2019
    # centerY_2019 = (N - cur_TN_2019) / (cur_BN_2019 - cur_TN_2019) * map_h_2019
    # # resolution of the satellite map
    # lit_per_pix_2019 = (cur_BE_2019 - cur_TE_2019) / map_w_2019
    # lat_per_pix_2019 = (cur_TN_2019 - cur_BN_2019) / map_h_2019
    # # compute the corresponding position of the big satellite image
    # centerX_2022 = (E - cur_TE_2022) / (cur_BE_2022 - cur_TE_2022) * map_w_2022
    # centerY_2022 = (N - cur_TN_2022) / (cur_BN_2022 - cur_TN_2022) * map_h_2022
    # # resolution of the satellite map
    # lit_per_pix_2022 = (cur_BE_2022 - cur_TE_2022) / map_w_2022
    # lat_per_pix_2022 = (cur_TN_2022 - cur_BN_2022) / map_h_2022

    # center crop and resize the UAV image
    croped_image = resize(image, target_size=(720, 540))

    # create target related dir
    # fileClassIndex = sixNumber(index)
    fileClassIndex = JPG.split("/")[-2:]
    fileClassIndex = "{}_{}_{}".format(place, fileClassIndex[0], fileClassIndex[1].split(".JPG")[0])

    fileClassDir = os.path.join(SiamUAV_train_dir, fileClassIndex)
    checkAndMkdir(fileClassDir)

    fileClassDirUAV = os.path.join(SiamUAV_train_dir, fileClassIndex, "UAV")
    checkAndMkdir(fileClassDirUAV)

    fileClassDirSatellite = os.path.join(SiamUAV_train_dir, fileClassIndex, "Satellite")
    checkAndMkdir(fileClassDirSatellite)

    # imwrite UAV image
    UAV_target_path = os.path.join(fileClassDirUAV, "0.JPG")
    cv2.imwrite(UAV_target_path, croped_image)

    # save uav GPS info
    GPS_info = {}
    GPS_info["UAV"] = {"E": E, "N": N}

    # crop the corresponding part of the satellite map
    mapsize = 4000
    # # target_size = 1280
    # croped_satellite_image = CenterCropFromSatellite(map_image_2019, Position=(centerX_2019, centerY_2019), mapsize=mapsize)
    # croped_satellite_image = CenterCropFromSatellite(map_image_2022, Position=(centerX_2022, centerY_2022), mapsize=mapsize)
    # Sat_topleft_E_2019 = E - mapsize // 2 * lit_per_pix_2019
    # Sat_topleft_N_2019 = N + mapsize // 2 * lat_per_pix_2019
    # Sat_bottomright_E_2019 = E + mapsize // 2 * lit_per_pix_2019
    # Sat_bottomright_N_2019 = N - mapsize // 2 * lat_per_pix_2019

    # open the json file for record the position of the UAV
    json_dict = {}
    GPS_info["Satellite"] = {}
    for i_repeat in range(SATELLITE_NUM_Repeat):
        repeat_list = list(range(map_range[0],map_range[1]+1,stride))
        for year in ["2019","2022"]:
            # compute the corresponding position of the big satellite image
            centerX = (E - info_dict[year]["cur_TE"]) / (info_dict[year]["cur_BE"] - info_dict[year]["cur_TE"]) * info_dict[year]["map_w"]
            centerY = (N - info_dict[year]["cur_TN"]) / (info_dict[year]["cur_BN"] - info_dict[year]["cur_TN"]) * info_dict[year]["map_h"]
            # resolution of the satellite map
            lit_per_pix = (info_dict[year]["cur_BE"] - info_dict[year]["cur_TE"]) / info_dict[year]["map_w"]
            lat_per_pix = (info_dict[year]["cur_TN"] - info_dict[year]["cur_BN"]) / info_dict[year]["map_h"]
            # crop the corresponding part of the satellite map
            croped_satellite_image = CenterCropFromSatellite(info_dict[year]["map_image"], Position=(centerX, centerY), mapsize=mapsize)
            Sat_topleft_E = E - mapsize // 2 * lit_per_pix
            Sat_topleft_N = N + mapsize // 2 * lat_per_pix
            # Sat_bottomright_E = E + mapsize // 2 * lit_per_pix
            # Sat_bottomright_N = N - mapsize // 2 * lat_per_pix
            for i,now_mapsize in enumerate(repeat_list):
                name = "{}_{}.jpg".format(year,i_repeat*len(repeat_list)+i)
                satellite_image, [X, Y], bbox, [center_distribute_X, center_distribute_Y],mapsize_ = random_crop_augmentation_function(croped_satellite_image, mapsize = now_mapsize)
                Loc_LX, Loc_LY, Loc_BX, Loc_BY = bbox
                new_lit_per_pix = lit_per_pix
                new_lat_per_pix = lat_per_pix
                Sat_topleft_E_new = Sat_topleft_E + Loc_LX * new_lit_per_pix
                Sat_topleft_N_new = Sat_topleft_N - Loc_LY * new_lat_per_pix
                Sat_bottomright_E_new = Sat_topleft_E + Loc_BX * new_lit_per_pix
                Sat_bottomright_N_new = Sat_topleft_N - Loc_BY * new_lat_per_pix

                Satellite_target_path = os.path.join(fileClassDirSatellite, name)
                satellite_image = cv2.resize(satellite_image,(Satellite_output_size,Satellite_output_size))
                X = int(X/mapsize_*Satellite_output_size)
                Y = int(Y/mapsize_*Satellite_output_size)
                # visualize
                # cropped satellite image
                # satellite_image = cv2.circle(satellite_image,(Y,X),10,color=(255,0,0),thickness=3)
                # cv2.imwrite("satellite.jpg",satellite_image)
                # # origin uav image
                # cv2.circle(croped_image,(np.array(croped_image.shape[:2][::-1])/2).astype(int),10,color=(255,0,0),thickness=3)
                # cv2.imwrite("uav.jpg",croped_image)
                #
                # map_img = map_image.copy()
                # # complete satellite image
                # left_x_cropped_satellite = int((Sat_topleft_E_new-cur_TE)/lit_per_pix)
                # left_y_cropped_satellite = int((cur_TN-Sat_topleft_N_new)/lat_per_pix)
                # right_x_cropped_satellite = int((Sat_bottomright_E_new-cur_TE)/lit_per_pix)
                # right_y_cropped_satellite = int((cur_TN-Sat_bottomright_N_new)/lat_per_pix)
                # complete_image = cv2.rectangle(map_img,(left_x_cropped_satellite,left_y_cropped_satellite),(right_x_cropped_satellite,right_y_cropped_satellite),color=[0,0,255],thickness=20)
                # complete_image = cv2.resize(complete_image,dsize=(map_w//2, map_h//2))
                # cv2.imshow("complete image",complete_image)
                #
                # cv2.waitKey(0)
                #
                # cv2.destroyWindow("complete image")

                # imwrite the satellite image to the target folder
                cv2.imwrite(Satellite_target_path, satellite_image)
                json_dict[name] = [X, Y]

                GPS_info["Satellite"][name] = {"tl_E": Sat_topleft_E_new, "tl_N": Sat_topleft_N_new,
                                            "br_E": Sat_bottomright_E_new, "br_N": Sat_bottomright_N_new,
                                            "center_distribute_X": center_distribute_X,"center_distribute_Y":center_distribute_Y,
                                            "map_size":mapsize_
                                            }

    # write the UAV position in the json file
    with open(os.path.join(fileClassDir, "labels.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

    with open(os.path.join(fileClassDir, "GPS_info.json"), 'w') as f:
        json.dump(GPS_info, f, indent=4, ensure_ascii=False)
    # index += 1

# source path
root = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/"
train_source_path = os.path.join(root, "oridata", "test")
# target path
SiamUAV_root = "/home/dmmm/Dataset/FPI/FPI2023"
SiamUAV_train_dir = os.path.join(SiamUAV_root, "2019_2022_satellitemap_700-1800_cr0.9_stride100")
checkAndMkdir(SiamUAV_train_dir)

dirlist = os.listdir(os.path.join(train_source_path,"University_UAV_Images"))
MapPath = [i + ".tif" for i in dirlist]
position_info_path_2019 = os.path.join(train_source_path,"old_tif", "PosInfo.txt")
position_info_path_2022 = os.path.join(train_source_path,"new_tif", "PosInfo.txt")
map_position_dict_2019 = getMapPosition(position_info_path_2019)
map_position_dict_2022 = getMapPosition(position_info_path_2022)

for i in range(len(MapPath)):
    place, map_ = dirlist[i], MapPath[i]
    dir = os.path.join(train_source_path, "University_UAV_Images",place)
    map_2019 = os.path.join(train_source_path, "old_tif", map_)
    map_2022 = os.path.join(train_source_path, "new_tif", map_)
    images_list = glob.glob(os.path.join(dir, "*/*.JPG"))
    # satellite map image
    map_image_2019 = cv2.imread(map_2019)
    map_h_2019, map_w_2019, _ = map_image_2019.shape
    map_image_2019 = cv2.resize(map_image_2019, (map_w_2019 // 2, map_h_2019 // 2), interpolation=3)
    map_h_2019, map_w_2019, _ = map_image_2019.shape
    cur_TN_2019, cur_TE_2019, cur_BN_2019, cur_BE_2019 = map_position_dict_2019[place]
    map_image_2022 = cv2.imread(map_2022)
    map_h_2022, map_w_2022, _ = map_image_2022.shape
    map_image_2022 = cv2.resize(map_image_2022, (map_w_2022 // 2, map_h_2022 // 2), interpolation=3)
    map_h_2022, map_w_2022, _ = map_image_2022.shape
    cur_TN_2022, cur_TE_2022, cur_BN_2022, cur_BE_2022 = map_position_dict_2022[place]
    info_dict = {
        "2019":{
            "map_image":map_image_2019,
            "map_h":map_h_2019,
            "map_w":map_w_2019,
            "cur_TN":cur_TN_2019,
            "cur_TE":cur_TE_2019,
            "cur_BN":cur_BN_2019,
            "cur_BE":cur_BE_2019,
        },
        "2022":{
            "map_image":map_image_2022,
            "map_h":map_h_2022,
            "map_w":map_w_2022,
            "cur_TN":cur_TN_2022,
            "cur_TE":cur_TE_2022,
            "cur_BN":cur_BN_2022,
            "cur_BE":cur_BE_2022
        }
    }
    # instance the augmentation function
    random_crop_augmentation_function = RandomCrop(cover_rate=0.9, map_size=map_range)
    p = Pool(10)
    for ind,res in enumerate(p.imap(pool_func, images_list)):
        # break
        if ind%100==0 and ind > 0:
            print("No.{} {}/{}".format(i,ind,len(images_list)))
    p.close()