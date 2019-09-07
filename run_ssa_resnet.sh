# This script is to run SSAResNet-18 for UCF101

python3 main.py --root_path /path_to/root/ --video_path /path_to/video_frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results_SSAresnet18 --dataset ucf101 --model SSAresnet --model_depth 18 --n_classes 101 --batch_size 8 --n_threads 4 --checkpoint 5 --resnet_shortcut A --learning_rate 0.1  --lr_patience 20

#--resume_path results_SSAresnet18/save_175.pth
# For SSAresnet with depth 18 and 34, resnet_shortcut is 'A'. For other depths, resnet_shortcut is 'B'
