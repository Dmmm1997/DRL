import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile, copytree, rmtree
from torch import nn
import logging
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format
import copy



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def copy_file_or_tree(path, target_dir):
    basename = os.path.basename(path)
    target_path = os.path.join(target_dir, basename)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path, target_path)
    elif os.path.isfile(path):
        copyfile(path, target_path)


def copyfiles2checkpoints(opt):
    dir_name = os.path.join('checkpoints', opt.name)
    if opt.debug != 1:
        if os.path.exists(dir_name):
            raise NameError(
                "{} 已经存在请更换name参数或者删除chekcpoint!!!".format(dir_name))
    os.makedirs(dir_name, exist_ok=True)
    # record every run
    copy_file_or_tree('train.py', dir_name)
    copy_file_or_tree('test_server.py', dir_name)
    copy_file_or_tree('test_meter.py', dir_name)
    copy_file_or_tree('evaluate_gpu.py', dir_name)
    copy_file_or_tree('evaluate_all.py', dir_name)
    copy_file_or_tree('datasets', dir_name)
    copy_file_or_tree('losses', dir_name)
    copy_file_or_tree('models', dir_name)
    copy_file_or_tree('optimizers', dir_name)
    copy_file_or_tree('tool', dir_name)
    copy_file_or_tree('train_test_local.sh', dir_name)
    copy_file_or_tree('heatmap.py', dir_name)
    copy_file_or_tree(opt.config, dir_name)


    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def save_network(network, dirname, epoch_label):
    target_dir = os.path.join("checkpoints", dirname, "output")
    os.makedirs(target_dir, exist_ok=True)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(target_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def check_box(images, boxes):
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    boxes = (boxes.cpu().detach().numpy()/16*255).astype(np.int)
    for img, box in zip(images, boxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        rect = plt.Rectangle(box[0:2], box[2]-box[0], box[3]-box[1])
        ax.add_patch(rect)
        plt.show()

def calc_flops_params(model,
                      input_size_drone,
                      input_size_satellite,
                      ):
    inputs_drone = torch.randn(input_size_drone).cuda()
    inputs_satellite = torch.randn(input_size_satellite).cuda()
    total_ops, total_params = profile(
        copy.deepcopy(model), (inputs_drone, inputs_satellite,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)


class TensorBoardManager:
    def __init__(self, init_path) -> None:
        os.makedirs(init_path, exist_ok=True)
        self.writer = SummaryWriter(init_path)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.writer.add_image(
            tag, img_tensor, global_step, walltime, dataformats)

    def add_images(self, tag, img_tensors, global_step=None, walltime=None, dataformats='NCHW'):
        self.writer.add_images(
            tag, img_tensors, global_step=global_step, walltime=walltime, dataformats=dataformats)
