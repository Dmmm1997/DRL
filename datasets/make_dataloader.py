import torch
from .SiamUAV import SiamUAVCenter,SiamUAV_val
import numpy as np



def make_dataset(opt,train=True):
    if train:
        image_datasets = SiamUAVCenter(opt)
        dataloaders =torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=opt.data_config.batchsize,
                                                 shuffle=True,
                                                 num_workers=opt.data_config.num_worker,
                                                 pin_memory=True,
                                                 # collate_fn=train_collate_fn
                                                 )
        dataset_sizes = {x: len(image_datasets) for x in ['satellite', 'drone']}
        return dataloaders, dataset_sizes

    else:
        dataset_test = SiamUAV_val(opt)
        dataloaders = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=opt.data_config.val_batchsize,
                                                  shuffle=False,
                                                  num_workers=opt.data_config.num_worker,
                                                  pin_memory=True)
        indices = np.random.choice(range(len(dataset_test)),2000)
        dataset_test_subset = torch.utils.data.Subset(dataset_test, indices)
        dataloaders_sub = torch.utils.data.DataLoader(dataset_test_subset,
                                                  batch_size=opt.data_config.val_batchsize,
                                                  shuffle=False,
                                                  num_workers=opt.data_config.num_worker,
                                                  pin_memory=True)
        return dataloaders, dataloaders_sub




