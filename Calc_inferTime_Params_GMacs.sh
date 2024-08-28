# file_list=("checkpoints/#Backbone/ConvNextT_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/CvT13_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/CvT21_CCN_SGF_Balance_cr1_nw15.py" 
# "checkpoints/#Backbone/DeiTS_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/EfficientNetB5_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/PcPvTS_CCN_SGF_Balance_cr1_nw15.py" 
# "checkpoints/#Backbone/PvTS_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/ResNet50_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/ViTB_CCN_SGF_Balance_cr1_nw15.py" 
# "checkpoints/#Backbone/ViTS_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Structure/CvT13_CCN_MCA_Balance_cr1_nw15.py" "checkpoints/#Structure/CvT13_CCN_MSA_Balance_cr1_nw15.py" 
# "checkpoints/#Structure/CvT13_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Structure/MixCvT13_CCN_AvgPool_Balance_cr1_nw15.py" "checkpoints/#Structure/MixCvT13_CCN_CE_Balance_cr1_nw15.py" 
# "checkpoints/#Structure/MixCvT13_CCN_GemPool_Balance_cr1_nw15.py" "checkpoints/#Structure/MixCvT13_CCN_SGF_Balance_cr1_nw15.py")


file_list=("checkpoints/#Backbone/ViTB_CCN_SGF_Balance_cr1_nw15.py" "checkpoints/#Backbone/ViTS_CCN_SGF_Balance_cr1_nw15.py")


for file in ${file_list[*]}; do
  IFS='/' read -ra parts <<< "$file"
  filename="${parts[-1]}"
  echo $filename
  cd $file
  python3 tool/get_inference_time.py --config $filename
  cd /home/dmmm/VscodeProject/FPI
done