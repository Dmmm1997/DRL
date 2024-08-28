import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from get_property import find_GPS_image
import json
from multiprocessing import Pool
from PIL import Image
SATELLITE_NUM = 20

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
    # LINEAR Interpolation
    if target_size:
        new_image = cv2.resize(new_image,target_size)

    return new_image

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


class RandomCrop(object):
    """
    random crop from satellite and return the changed label
    """
    def __init__(self, cover_rate=0.9,map_size=(512,1000)):
        """
        map_size: (low_size,high_size)
        cover_rate: the max cover rate
        """
        self.cover_rate = cover_rate
        self.map_size = map_size



    def __call__(self, img):
        map_size = np.random.randint(int(self.map_size[0]),int(self.map_size[1]))
        if map_size%2==1:
            map_size-=1
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h,w,c = img.shape
        cx,cy = h//2,w//2
        # bbox = np.array([cx-self.map_size//2,cy-self.map_size//2,cx+self.map_size//2,cy+self.map_size//2],dtype=np.int)
        # new_map = img[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1,:]
        # assert new_map.shape[0:2] == [self.map_size,self.map_size], "the size is not correct"
        RandomCenterX = np.random.randint(int(0.5*h-self.cover_rate/2*map_size),int(0.5*h+self.cover_rate/2*map_size))
        RandomCenterY = np.random.randint(int(0.5*w-self.cover_rate/2*map_size),int(0.5*w+self.cover_rate/2*map_size))
        bbox = np.array([RandomCenterX-map_size//2,
                         RandomCenterY-map_size//2,
                         RandomCenterX+map_size//2,
                         RandomCenterY+map_size//2],dtype=int)

        bias_left = bias_top = bias_right = bias_bottom = 0
        if bbox[0] < 0:
            bias_top = int(-1 * bbox[0])
            bbox[0] = 0
        if bbox[1] < 0:
            bias_left = int(-1 * bbox[1])
            bbox[1] = 0
        if bbox[2] >= w:
            bias_bottom = int(bbox[2] - h)
            bbox[2] = w
        if bbox[3] >= h:
            bias_right = int(bbox[3] - w)
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
        image = Image.fromarray(croped_image.astype('uint8')).convert('RGB')
        return image,[ratex,ratey]


def randomCropSatelliteMap(Mapimage, Position, cover_rate, mapsize ,target_size):
    '''
    random crop the part in the satellite map
    :param Mapimage: ori big satellite map
    :param Position: the UAV position in the big satellite image (CenterX,CenterY)
    :param cover_rate: the rate of the random position cover the map size
    :param mapsize: the UAV corresponding size in the satellite map
    :param target_size: the final output size of the satellite image
    :return: cropped image and the UAV image's position in the cropped image
    '''
    h,w,c = Mapimage.shape
    centerX,centerY = Position
    center_cover_pixels = cover_rate*mapsize
    centerX_changed_Min = centerX-center_cover_pixels//2
    centerX_changed_Max = centerX+center_cover_pixels//2
    centerY_changed_Min = centerY-center_cover_pixels//2
    centerY_changed_Max = centerY+center_cover_pixels//2
    random_value_X = np.random.randint(centerX_changed_Min,centerX_changed_Max)
    random_value_Y = np.random.randint(centerY_changed_Min,centerY_changed_Max)
    # put the X and Y between in the Mapimage
    random_value_X = max(0,min(w-1,random_value_X))
    random_value_Y = max(0,min(h-1,random_value_Y))
    # get bbox [x1,y1,x2,y2]
    bbox = [random_value_X-mapsize//2,random_value_Y-mapsize//2,random_value_X+mapsize//2,random_value_Y+mapsize//2]
    bias_left = bias_top = bias_right = bias_bottom = 0
    if bbox[0]<0:
        bias_left = -1*bbox[0]
        bbox[0] = 0
    if bbox[1]<0:
        bias_top = -1*bbox[1]
        bbox[1] = 0
    if bbox[2]>=w:
        bias_right = bbox[2]-w
        bbox[2] = w
    if bbox[3]>=h:
        bias_bottom = bbox[3]-h
        bbox[3] = h
    croped_image = Mapimage[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    mean = np.mean(croped_image,axis=(0,1),dtype=float)
    # padding the border
    croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT, value=mean)

    target_image = cv2.resize(croped_image,target_size)

    # get the bias of the center
    bias_center_x = centerX-random_value_X
    bias_center_y = centerY-random_value_Y
    rate_of_position_x = 0.5+bias_center_x/mapsize
    rate_of_position_y = 0.5+bias_center_y/mapsize

    return target_image, [rate_of_position_x,rate_of_position_y]


def mutiProcess(class_file):
    index = 0
    dir, map = dirlist[class_file], MapPath[class_file]
    images_list = glob.glob(os.path.join(dir, "*/*.JPG"))
    # place name
    place = dir.split("/")[-1]
    # satellite map image
    map_image = cv2.imread(map)
    map_h_, map_w_, _ = map_image.shape
    map_image = cv2.resize(map_image,(map_w_//2,map_h_//2), interpolation=3)
    map_h, map_w, _ = map_image.shape
    cur_TN, cur_TE, cur_BN, cur_BE = map_position_dict[place]
    # instance the augmentation function
    random_crop_augmentation_function = RandomCrop(cover_rate=0.9,map_size=(512,1000))
    for JPG in tqdm(images_list):
        # image
        image = cv2.imread(JPG)
        uav_h, uav_w, _ = image.shape
        # position info
        GPS_info = find_GPS_image(JPG)
        y = list(list(GPS_info.values())[0].values())
        E, N = y[3], y[1]
        # compute the corresponding position of the big satellite image
        centerX = (E - cur_TE) / (cur_BE - cur_TE) * map_w
        centerY = (N - cur_TN) / (cur_BN - cur_TN) * map_h
        # center crop and resize the UAV image
        croped_image = center_crop_and_resize(image, target_size=(256, 256))
        # create target related dir
        fileClassIndex = sixNumber(index)
        fileClassIndex = "{}_{}".format(place,fileClassIndex)

        fileClassDir = os.path.join(SiamUAV_train_dir, fileClassIndex)
        checkAndMkdir(fileClassDir)

        fileClassDirUAV = os.path.join(SiamUAV_train_dir, fileClassIndex, "UAV")
        checkAndMkdir(fileClassDirUAV)

        fileClassDirSatellite = os.path.join(SiamUAV_train_dir, fileClassIndex, "Satellite")
        checkAndMkdir(fileClassDirSatellite)

        # open the json file for record the position of the UAV
        json_dict = {}

        # imwrite UAV image
        UAV_target_path = os.path.join(fileClassDirUAV, "0.JPG")
        cv2.imwrite(UAV_target_path, croped_image)
        # crop the corresponding part of the satellite map
        for i in range(SATELLITE_NUM):
            croped_satellite_image, [ratex, ratey] = randomCropSatelliteMap(map_image, Position=(centerX, centerY),
                                                                            cover_rate=0.7, mapsize=1200,
                                                                            target_size=(512, 512))
            # visualize the localization result

            # h,w,_ = croped_satellite_image.shape
            # croped_satellite_image = cv2.circle(croped_satellite_image,(int(ratex*w),int(ratey*h)),10,color=(255,0,0))
            # cv2.imshow("2",croped_image)
            # cv2.imshow("1",croped_satellite_image)
            # cv2.waitKey(0)

            # imwrite satellite image
            Satellite_target_path = os.path.join(fileClassDirSatellite, "{}.tif".format(i))
            cv2.imwrite(Satellite_target_path, croped_satellite_image)

            json_dict["{}.tif".format(i)] = [ratex, ratey]

        # write the UAV position in the json file
        with open(os.path.join(fileClassDir, "labels.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, ensure_ascii=False)

        index += 1

if __name__ == '__main__':
    # source path
    root = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/"
    train_source_path = os.path.join(root, "oridata", "test")
    # target path
    SiamUAV_root = "/home/dmmm/SiamUAV/"
    SiamUAV_train_dir = os.path.join(SiamUAV_root,"SiamUAV_Test")
    checkAndMkdir(SiamUAV_train_dir)

    dirlist = [i for i in glob.glob(os.path.join(train_source_path, "*")) if os.path.isdir(i)]
    MapPath = [i + ".tif" for i in dirlist]
    for i in MapPath:
        if not os.path.exists(i):
            raise NameError("name is not corresponding!")
    position_info_path = os.path.join(train_source_path, "PosInfo.txt")
    map_position_dict = getMapPosition(position_info_path)

    # muti-process
    P = Pool(processes=4)
    P.map(func=mutiProcess, iterable=range(len(MapPath)))

    # single-process
    # for i in range(len(MapPath)):
    #     mutiProcess(i)