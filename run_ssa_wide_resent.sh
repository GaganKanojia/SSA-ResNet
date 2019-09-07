# This script is to run SSAWide_ResNet for UCF101

python3 main.py --root_path /path_to/root/ --video_path /path_to/video_frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results_SSAwide_resnet50 --dataset ucf101 --model SSAwideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2 --n_classes 101 --batch_size 8 --n_threads 4 --checkpoint 1 --learning_rate 0.1  --lr_patience 20

#--resume_path results_ours_3d18/save_45.pth


