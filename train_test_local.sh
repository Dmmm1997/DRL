config="configs/##target_config/MixCvT13_FPN128_outstride1nearest_CE_Balance_cr31.py"
gpu_ids=0
name="${config#*/}"
python train.py --config $config --name $name

# mode="2019_2022_satellitemap_700-1800_cr0.9_stride100"
cd checkpoints/$name
test_config="${config#*/*/}"
echo $test_config
python test_meter.py --config $test_config
cd /home/dmmm/VscodeProject/FPI

