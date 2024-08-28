import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import shutil


mode = "test"
root_dir = "/home/dmmm/Dataset/FPI/FPI2023"
target_dir = "/home/dmmm/Dataset/FPI/FPI2023_S"
mode_dir = os.path.join(root_dir,mode)
target_mode_dir = os.path.join(target_dir,mode)
os.makedirs(target_mode_dir,exist_ok=True)

def image_process(image):
    img = cv2.imread(image)
    Place_ID,UAV_Satellite_type,basename = image.split("/")[-3:]
    new_basename = basename.split(".")[0]+".jpg"
    tmp_dir = os.path.join(target_mode_dir,Place_ID,UAV_Satellite_type)
    os.makedirs(tmp_dir,exist_ok=True)
    cv2.imwrite(os.path.join(tmp_dir,new_basename),img)


def file_process(file):
    Place_ID,basename = file.split("/")[-2:]
    new_filename = os.path.join(target_mode_dir,Place_ID,basename)
    shutil.copyfile(file,new_filename)


p = Pool(10)
if mode == "train":
    satellite_images = glob.glob(os.path.join(mode_dir,"*/*/*.tif"))
    uav_images = glob.glob(os.path.join(mode_dir,"*/*/*.JPG"))
    total_images = satellite_images+uav_images
    for res in tqdm(p.imap(image_process,total_images)):
        pass
if mode in ["test", "val"]:
    satellite_images = glob.glob(os.path.join(mode_dir,"*/*/*.jpg"))
    uav_images = glob.glob(os.path.join(mode_dir,"*/*/*.JPG"))
    
    total_images = satellite_images+uav_images
    for res in tqdm(p.imap(image_process,total_images)):
        pass

    json_files = glob.glob(os.path.join(mode_dir,"*/*.json"))
    for res in tqdm(p.imap(file_process,json_files)):
        pass

p.close()