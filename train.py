# -*- coding: utf-8 -*-
import argparse
import torch

from torch.autograd import Variable
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from torch.cuda.amp import autocast
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from losses.make_loss import make_loss
from tool.utils_server import calc_flops_params, save_network, copyfiles2checkpoints, get_logger, TensorBoardManager
from tool.evaltools import evaluate
from tqdm import tqdm
import numpy as np
import cv2
import random
import os
import json
from collections import defaultdict
from tool.evaltools import Distance
from mmcv import Config
import datetime


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def get_config():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--config', default='configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py',
        type=str, help='config filename')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default="test",
                        type=str, help='output model name')
    opt = parser.parse_args()
    if opt.name == "":
        opt.name = opt.config.split("/")[-1].split(".py")[0].split("configs/")[-1]
    print(opt.name)
    cfg = Config.fromfile(opt.config)
    for key, value in cfg.items():
        setattr(opt, key, value)
    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)

def setup_device(opt):
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        # cudnn.benchmark = True


def train_model(model, loss_func, opt, dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.train_config["num_epochs"]
    output_dir = os.path.join("checkpoints", opt.name, "output")
    os.makedirs(output_dir, exist_ok=True)
    cur_time = datetime.datetime.now()
    logger_file = os.path.join(output_dir, "train_{}.log".format(cur_time))
    logger = get_logger(logger_file)
    # init tensorboard writer
    tensorboard_writer = TensorBoardManager(
        os.path.join(output_dir, "summary"))
    
    macs, params = calc_flops_params(
        model, (1, 3, opt.data_config['UAVhw'][0], opt.data_config['UAVhw'][1]), (1, 3, opt.data_config['Satellitehw'][0], opt.data_config['Satellitehw'][1]))
    logger.info("MACs={}, Params={}".format(macs, params))

    since = time.time()

    scaler = GradScaler()

    best_RDS = 0

    logger.info('start training!')

    optimizer, scheduler = make_optimizer(model, opt)

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        logger.info('-' * 50)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        iter_cls_loss = 0.0
        iter_loc_loss = 0.0
        iter_start = time.time()
        iter_loss = 0
        total_iters = len(dataloaders["train"])
        # train
        for iter, (z, x, ratex, ratey) in enumerate(dataloaders["train"]):
            now_batch_size, _, _, _ = z.shape

            if now_batch_size < opt.data_config["batchsize"]:  # skip the last batch
                continue
            if use_gpu:
                z = Variable(z.cuda().detach())
                x = Variable(x.cuda().detach())
            else:
                z, x = Variable(z), Variable(x)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # start_time = time.time()
            # if opt.train_config["autocast"]:
            #     with autocast():
            #         outputs = model(z, x)  # satellite and drone
            # else:
            outputs = model(z, x)
            # print("model_time:{}".format(time.time()-start_time))
            cls_loss, loc_loss = loss_func(outputs, [ratex, ratey])
            loss = cls_loss + loc_loss
            # backward + optimize only if in training phase
            loss_backward = loss
            # start_time = time.time()
            if opt.train_config["autocast"]:
                scaler.scale(loss_backward).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                loss_backward.backward()
                optimizer.step()
                scheduler.step()
            # print("loss_backward_time:{}".format(time.time()-start_time))

            # statistics
            running_loss += loss.item() * now_batch_size
            iter_loss += loss.item() * now_batch_size
            iter_cls_loss += cls_loss.item() * now_batch_size
            iter_loc_loss += loc_loss.item() * now_batch_size

            if (iter + 1) % opt.log_interval == 0:
                time_elapsed_part = time.time() - iter_start
                iter_loss = iter_loss/opt.log_interval/now_batch_size
                iter_cls_loss = iter_cls_loss/opt.log_interval/now_batch_size
                iter_loc_loss = iter_loc_loss/opt.log_interval/now_batch_size

                lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

                tensorboard_writer.add_scalar(
                    "loss/total_loss", iter_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "loss/cls_loss", iter_cls_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "loss/loc_loss", iter_loc_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "lr", lr_backbone, epoch*total_iters+iter)

                logger.info("[{}/{}] loss: {:.4f} cls_loss: {:.4f} loc_loss:{:.4f} lr_backbone:{:.6f} time:{:.0f}m {:.0f}s ".format(
                    iter + 1, total_iters, iter_loss, iter_cls_loss, iter_loc_loss, lr_backbone, time_elapsed_part // 60, time_elapsed_part % 60))
                iter_loss = 0.0
                iter_loc_loss = 0.0
                iter_cls_loss = 0.0
                iter_start = time.time()

        epoch_loss = running_loss / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

        time_elapsed = time.time() - since
        logger.info('Epoch[{}/{}] Loss: {:.4f}  lr_backbone:{:.6f}  time:{:.0f}m {:.0f}s'.format(
            epoch+1, num_epochs, epoch_loss, lr_backbone, time_elapsed // 60, time_elapsed % 60))

        # ----------------------save and test the model------------------------------ #
        if ((epoch + 1)-opt.checkpoint_config["epoch_start_save"]) % opt.checkpoint_config["interval"] == 0 and (epoch+1) >= opt.checkpoint_config["epoch_start_save"] or (epoch+1 == opt.train_config["num_epochs"]):
            # if "only_save_best" is False， save the checkpoint
            if not opt.checkpoint_config["only_save_best"]:
                save_name = "last" if epoch+1 == opt.train_config["num_epochs"] else epoch+1
                save_network(model, opt.name, save_name)
            model.eval()
            total_score = 0.0
            total_score_b = 0.0
            start_time = time.time()
            flag_bias = 0
            MA_json_save = []
            MA_dict = defaultdict(int)
            MA_log_list = [1, 3, 5, 10, 20, 30, 50, 100]
            tensorboard_image_ind = 0
            val_loss = 0.0
            sample_nums = 0
            for uav, satellite, X, Y, uav_path, satellite_path in tqdm(dataloaders["val"]):
                sample_nums += uav.shape[0]
                z = uav.cuda()
                x = satellite.cuda()
                rate_x = X/opt.data_config["Satellitehw"][0]
                rate_y = Y/opt.data_config["Satellitehw"][1]
                with torch.no_grad():
                    response, loc_bias = model(z, x)
                    cls_loss, loc_loss = loss_func([response, loc_bias], [rate_x, rate_y])
                val_iter_loss = cls_loss + loc_loss
                val_loss += val_iter_loss/len(dataloaders["val"])

                if opt.model["loss"]["cls_loss"].get("use_softmax", False):
                    response = torch.softmax(response,dim=1)[:,1:]
                else:
                    response = torch.sigmoid(response)
                maps = response.squeeze().cpu().detach().numpy()
                # 遍历每一个batch
                for ind, map in enumerate(maps):
                    if opt.test_config["filterR"] != 1:
                        kernel = create_hanning_mask(opt.test_config["filterR"])
                        map = cv2.filter2D(map, -1, kernel)

                    label_XY = np.array(
                        [X[ind].squeeze().detach().numpy(), Y[ind].squeeze().detach().numpy()])

                    satellite_map = cv2.resize(map, opt.data_config["Satellitehw"])
                    id = np.argmax(satellite_map)
                    S_X = int(id // opt.data_config["Satellitehw"][0])
                    S_Y = int(id % opt.data_config["Satellitehw"][1])

                    # 获取预测的经纬度信息
                    get_gps_x = S_X / opt.data_config["Satellitehw"][0]
                    get_gps_y = S_Y / opt.data_config["Satellitehw"][0]
                    path = satellite_path[ind].split("/")
                    read_gps = json.load(
                        open(
                            os.path.join(
                                satellite_path[ind].split("/Satellite")[0],
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
                    MA_json_save.append(meter_distance)
                    for meter in MA_log_list:
                        if meter_distance <= meter:
                            MA_dict[meter] += 1

                    # 统计RDS指标
                    pred_XY = np.array([S_X, S_Y])
                    single_score = evaluate(
                        opt, pred_XY=pred_XY, label_XY=label_XY)
                    total_score += single_score
                    if loc_bias is not None:
                        flag_bias = 1
                        loc = loc_bias.squeeze().cpu().detach().numpy()
                        id_map = np.argmax(map)
                        S_X_map = int(id_map // map.shape[-1])
                        S_Y_map = int(id_map % map.shape[-1])
                        pred_XY_map = np.array([S_X_map, S_Y_map])
                        pred_XY_b = (
                            pred_XY_map + loc[:, S_X_map, S_Y_map]) * opt.data_config["Satellitehw"][0] / loc.shape[-1]  # add bias
                        pred_XY_b = np.array(pred_XY_b)
                        single_score_b = evaluate(
                            opt, pred_XY=pred_XY_b, label_XY=label_XY)
                        total_score_b += single_score_b

                    # print("pred: " + str(pred_XY) + " label: " +str(label_XY) +" score:{}".format(single_score))
                    # TODO:将可视化图像添加到tensorboard中

            # time
            time_consume = time.time() - start_time
            logger.info("time consume is {}".format(time_consume))

            # total loss
            logger.info("valset total loss is {}".format(val_loss))

            # RDS
            RDS = total_score / sample_nums
            # save the best checkpoint
            if RDS > best_RDS:
                best_RDS = RDS
                best_epoch = epoch+1
                save_network(model, opt.name, "best")
            logger.info("Epoch{}: the RDS is {}".format(epoch+1, RDS))
            if flag_bias:
                RDS_b = total_score_b / sample_nums
                logger.info(
                    "Epoch{}: the bias RDS is {}".format(epoch+1, RDS_b))

            # MA@K
            for log_meter in MA_log_list:
                logger.info("MA@{}m = {:.4f}".format(log_meter,
                            MA_dict[log_meter]/sample_nums))
        else:
            val_loss = 0
            for uav, satellite, X, Y, uav_path, satellite_path in tqdm(dataloaders["val_sub"]):
                z = uav.cuda()
                x = satellite.cuda()
                rate_x = X/opt.data_config["Satellitehw"][0]
                rate_y = Y/opt.data_config["Satellitehw"][1]
                with torch.no_grad():
                    response, loc_bias = model(z, x)
                    cls_loss, loc_loss = loss_func([response, loc_bias], [rate_x, rate_y])
                val_iter_loss = cls_loss + loc_loss
                val_loss += val_iter_loss/len(dataloaders["val_sub"])
            # total loss
            logger.info("valset total loss is {}".format(val_loss))
    logger.info("saved best epoch is {}, RDS is {:.3f}".format(best_epoch, best_RDS))





if __name__ == '__main__':
    opt = get_config()

    # init device
    setup_device(opt)

    # init seed
    setup_seed(opt.seed)
    
    # init dataloader
    dataloaders_train, dataset_sizes = make_dataset(opt)
    dataloaders_val, dataloaders_val_sub = make_dataset(opt, train=False)
    dataloaders = {"train": dataloaders_train,
                   "val": dataloaders_val,
                   "val_sub": dataloaders_val_sub}
    opt.train_iters_per_epoch = len(dataloaders["train"])

    # init model
    model = make_model(opt)
    model = model.cuda()
    
    # init loss
    loss_func = make_loss(opt)
    # copy current demos to a seperate dir
    copyfiles2checkpoints(opt)
    # train the model
    train_model(model, loss_func, opt, dataloaders, dataset_sizes)
