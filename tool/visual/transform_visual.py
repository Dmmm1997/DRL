import sys
sys.path.append("./")
from datasets.Augmentation import RotateAndCrop, RandomErasing, RandomRotate90
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import os


def get_parse():
    parser = argparse.ArgumentParser(description='Transfrom Visualization')
    parser.add_argument(
        '--image_path', default='tool/visual/original/Jiliang_000111/UAV/0.JPG', type=str, help='')
    parser.add_argument(
        '--target_dir', default='tool/visual/results/transform/RandomRotate90', type=str, help='')
    parser.add_argument(
        '--num_aug', default=10, type=int, help='')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = get_parse()
    image = Image.open(opt.image_path)

    re = RandomErasing(probability=1.0)

    rr = RandomRotate90()
    # ra = transforms.RandomAffine(180)
    rac = RotateAndCrop(rate=1.0)
    cj = transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)

    os.makedirs(opt.target_dir, exist_ok=True)

    for ind in range(opt.num_aug):
        image_ = rr(image)
        image_ = np.array(image_)
        image_ = image_[:, :, [2, 1, 0]]
        h, w = image_.shape[:2]
        image_ = cv2.circle(
            image_.copy(), (int(w/2), int(h/2)), 3, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(opt.target_dir, "{}.jpg".format(ind)), image_)
