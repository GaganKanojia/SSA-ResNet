# This script is to run SSAResNeXt8 for modelnet10 with 12 rotation augmentations

python3 main.py --root_path /path_to/root/ --video_path modelnet10x12 --result_path results_SSAresnext8_modelnet10x12 --dataset modelnet --model SSAresnet_modelnet --model_depth 8 --batch_size 32 --checkpoint 1 --resnet_shortcut B --n_threads 0 --learning_rate 0.00001 

#--pretrain_path results_SSAresnext8_modelnet40x12/save_20.pth --n_classes 40 --n_finetune_classes 10

#--resume_path results_SSAresnext8_modelnet10x12/save_4.pth

