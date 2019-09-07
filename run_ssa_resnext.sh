# This script is to run SSAResNeXt-50 for UCF101

python3 main.py  --root_path /path_to/root/ --video_path /path_to/video_frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results_SSAresnext50 --dataset ucf101 --model SSAresnext --model_depth 50 --resnet_shortcut B --resnext_cardinality 32 --n_classes 101 --batch_size 8 --n_threads 4 --checkpoint 1 --learning_rate 0.1  --lr_patience 20

#--resume_path results_SSAresnext50/save_45.pth

