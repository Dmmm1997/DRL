import glob
from PIL import Image
from models.taskflow import make_model
from torchvision import transforms
import os
import numpy as np
import json
import torch
import argparse
import cv2
from datasets.Augmentation import RandomCrop

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--UAVhw', default=[112,112], type=int, help='')
    parser.add_argument('--Satellitehw', default=[288,288], type=int, help='')
    opt = parser.parse_args()
    return opt

opt = get_parse()
model = make_model(opt)
state_dict = torch.load("/home/dmmm/PycharmProject/SiamUAV/checkpoints/random_erase_centerdata_CenterMaskLoss-R=1/net_049.pth")
model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

transform_uav = transforms.Compose([
        transforms.Resize(opt.UAVhw, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_satellite = transforms.Compose([
    transforms.Resize(opt.Satellitehw, interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

random_crop = RandomCrop(cover_rate=0.7,map_size=(800,1000))
for i in range(100):
    test_root = "/media/dmmm/4T-3/DataSets/SiamUAV/test_center"
    videos = os.listdir(test_root)
    video = np.random.choice(videos)
    video_path = os.path.join(test_root,video)

    UAV_path = os.path.join(video_path,"UAV","0.JPG")
    UAV_image = Image.open(UAV_path)
    Satellite_path = glob.glob(os.path.join(video_path,"Satellite","*"))
    Satellite_random_path = np.random.choice(Satellite_path)
    basename = os.path.basename(Satellite_random_path)
    # ratex,ratey = json_context[basename]
    Satellite_image = Image.open(Satellite_random_path)
    Satellite_image,[ratex,ratey] = random_crop(Satellite_image)


    z = transform_uav(UAV_image).unsqueeze(0).cuda()
    x = transform_satellite(Satellite_image).unsqueeze(0).cuda()


    response = model(z,x)
    map = response.squeeze().cpu().detach().numpy()
    h,w = map.shape
    print(np.argmax(map)//h,np.argmax(map)%w)
    UAV_image = cv2.cvtColor(np.asarray(UAV_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("22",UAV_image)
    Satellite_image = Satellite_image.resize(opt.Satellitehw)
    Satellite_image = cv2.cvtColor(np.asarray(Satellite_image), cv2.COLOR_RGB2BGR)
    print(ratex*w,ratey*h)
    Satellite_image = cv2.circle(Satellite_image,(int(np.argmax(map)%w/w*opt.Satellitehw[1]),int(np.argmax(map)//h/h*opt.Satellitehw[0])),8,color=(255,0,0),thickness=3)
    Satellite_image = cv2.circle(Satellite_image,(int(ratey*opt.Satellitehw[1]),int(ratex*opt.Satellitehw[0])),8,color=(0,0,255),thickness=3) # x是宽 y是高
    cv2.imshow("11",Satellite_image)
    cv2.waitKey(0)
    print(np.argmax(map))




