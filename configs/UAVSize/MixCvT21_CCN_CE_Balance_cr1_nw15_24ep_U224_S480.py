# 1. 模型配置(models) =========================================
model = dict(
    backbone=dict(
        type="MixFormer",
        vit_type="cvt21",
        pretrain_path="/home/dmmm/VscodeProject/FPI/pretrain_model/CvT-21-384x384-IN-22k.pth",
        pretrain=True,
        output_index=[0],
    ),
    neck=dict(
        type="None",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=0,
    ),
    head=dict(
        type="ChannelEmbedding", input_ndim=384, mid_process_channels=[128, 32, 8, 1]
    ),
    loss=dict(
        cls_loss=dict(type="BalanceLoss", center_R=1, neg_weight=15),
        # reg_loss=dict(
        # )
    ),
)


# 2. 数据集配置(datasets) =========================================
data_config = dict(
    batchsize=4,
    num_worker=8,
    train_dir="/data/datasets/crossview/FPI/map2019/train",
    val_dir="/data/datasets/crossview/FPI/map2019/val",
    test_dir="/data/datasets/crossview/FPI/map2019",
    test_mode="merge_test_700-1800_cr0.95_stride100",
    UAVhw=[224, 224],
    Satellitehw=[480, 480],
)

pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            # RotateAndCrop=dict(rate=0.5),
            # RandomAffine=dict(degrees=180),
            # ColorJitter=dict(brightness=0.5, contrast=0.1, saturation=0.1, hue=0),
            # RandomErasing=dict(probability=0.3),
            RandomResize=dict(img_size=data_config["UAVhw"]),
            ToTensor=dict(),
        ),
        Satellite=dict(
            # ColorJitter=dict(brightness=0.5, contrast=0.1, saturation=0.1, hue=0),
            # RandomErasing=dict(probability=0.3),
            RandomResize=dict(img_size=data_config["Satellitehw"]),
            ToTensor=dict(),
        ),
    ),
)

# 3. 训练策略配置(schedules) =========================================
lr_config = dict(lr=2e-5, type="cosine", warmup_iters=0, warmup_ratio=0.001)
train_config = dict(autocast=True, num_epochs=24)
test_config = dict(
    num_worker=8,
    filterR=1,
    checkpoint="output/net_best.pth",
)

# 4. 运行配置(runtime) =========================================
checkpoint_config = dict(
    interval=2,
    epoch_start_save=12,
    only_save_best=True,
)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666