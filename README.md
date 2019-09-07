# Exploring Temporal Differences in 3D Convolutional Neural Networks

The code is for the paper titled as "Exploring Temporal Differences in 3D Convolutional Neural Networks". It is written in Pytorch. The major part of the source code is taken form [here](https://github.com/kenshohara/3D-ResNets-PyTorch)<br />

### Requirements
-Python (The code is tested with verison 3.5)<br />
-[Pytorch](https://pytorch.org/) (The code is tested with verison 1.1.0)<br />
-Numpy (The code is tested with verison 1.16.0)<br />
-CUDA (The code is tested with verison 9.0)<br />

For the preparation of the dataset, please refer to [this](https://github.com/kenshohara/3D-ResNets-PyTorch) GitHub repository.


To train SSA-ResNet (18 layers), use the following script<br />
```
python3 main.py --root_path /path_to/root/ --video_path /path_to/video_frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results_SSAresnet18 --dataset ucf101 --model SSAresnet --model_depth 18 --n_classes 101 --batch_size 8 --n_threads 4 --checkpoint 5 --resnet_shortcut A --learning_rate 0.1  --lr_patience 20 <br />
```
To resume the code from an intermediate result use --resume_path results_SSAresnet18/file_name.pth<br />

The scripts are provided with the source code to train the other models. For more info, refer [here](https://github.com/kenshohara/3D-ResNets-PyTorch)
