# This script is to run SSAResNeXt8 for modelnet40 with 12 rotation augmentations

python3 main.py --root_path /path_to/root/ --video_path modelnet40x12 --result_path results_SSAresnext8_modelnet40x12 --dataset modelnet --model SSAresnet_modelnet --model_depth 8 --n_classes 40 --batch_size 2 --checkpoint 1 --resnet_shortcut B --n_threads 0  --learning_rate 0.1  --sample_duration 32 --nesterov

#--resume_path results_SSAresnext8_modelnet10x12/save_10.pth 

