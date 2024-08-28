# 1. 模型配置(models) =========================================
model = dict(
    backbone=dict(type="ViT-S", pretrain=True, output_index=[0]),
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=0,
    ),
    head=dict(
        type="SingleGroupFusionHead",
        input_ndim=128,
        mid_ndim=128,
        num_classes=1,
        padding=True,
        enable_reg=False,
    ),
    postprocess=dict(
        upsample_to_original=False,
    ),
    loss=dict(
        cls_loss=dict(
            type="BalanceLoss",
            center_R=1,
            neg_weight=15,
            weight_rate=1.0,
            use_softmax=False,
        ),
    ),
)


# 2. 数据集配置(datasets) =========================================
data_config = dict(
    batchsize=8,
    num_worker=8,
    val_batchsize=8,
    train_dir="/data/datasets/crossview/FPI/map2019/train",
    # val_dir='/data/datasets/crossview/FPI/map2019/val',
    val_dir="/data/datasets/crossview/FPI/map2019/merge_test_700-1800_cr0.95_stride100",
    test_dir="/data/datasets/crossview/FPI/map2019",
    test_mode="merge_test_700-1800_cr0.95_stride100",
    UAVhw=[128, 128],
    Satellitehw=[384, 384],
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
lr_config = dict(lr=1.5e-4, type="cosine", warmup_iters=500, warmup_ratio=0.1)
train_config = dict(autocast=True, num_epochs=12)
test_config = dict(
    num_worker=8,
    filterR=1,
    checkpoint="output/net_best.pth",
)

# 4. 运行配置(runtime) =========================================
checkpoint_config = dict(
    interval=1,
    epoch_start_save=6,
    only_save_best=True,
)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666