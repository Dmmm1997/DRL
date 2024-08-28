import os
import glob
import numpy as np
import shutil
from tqdm import tqdm


test_dir = "/data/datasets/crossview/FPI/map2019/merge_test_700-1800_cr0.95_stride100"
test_pair_list = glob.glob(test_dir + "/*")

target_dir = "/data/datasets/crossview/FPI/map2019/val"
rate = 0.15

np.random.shuffle(test_pair_list)
test_pair_list = test_pair_list[: int(rate * len(test_pair_list))]


for pair in tqdm(test_pair_list):
    name = os.path.basename(pair)
    shutil.copytree(pair, os.path.join(target_dir, name))
