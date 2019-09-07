from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.modelnet import modelnet
import numpy as np

def get_training_set(opt, spatial_transform, temporal_transform,
					 target_transform):
	assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51','modelnet']

	if opt.dataset == 'kinetics':
		training_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			'training',n_samples_for_each_video= opt.samples_per_video,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'activitynet':
		training_data = ActivityNet(
			opt.video_path,
			opt.annotation_path,
			'training',
			False,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'ucf101':
		training_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			'training',
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'hmdb51':
		training_data = HMDB51(
			opt.video_path,
			opt.annotation_path,
			'training',
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform)
	elif opt.dataset == 'modelnet':
		if opt.video_path == 'modelnet10x12':
			data = np.load(opt.root_path+'modelnet10_train12_32.npz')
		elif opt.video_path == 'modelnet40x12':
			data = np.load(opt.root_path+'modelnet40_train12_32.npz')
		#data = np.load(opt.root_path+'modelnet10_train_32.npz')
		training_data = modelnet(data)

	return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
	assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51','modelnet']

	if opt.dataset == 'kinetics':
		validation_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'activitynet':
		validation_data = ActivityNet(
			opt.video_path,
			opt.annotation_path,
			'validation',
			False,
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'ucf101':
		validation_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'hmdb51':
		validation_data = HMDB51(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'modelnet':
		if opt.video_path == 'modelnet10x12':
			data = np.load(opt.root_path+'modelnet10_test_32.npz')
		elif opt.video_path == 'modelnet40x12':
			data = np.load(opt.root_path+'modelnet40_test_32.npz')

		validation_data = modelnet(data)

	return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
	assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
	assert opt.test_subset in ['val', 'test']

	if opt.test_subset == 'val':
		subset = 'validation'
	elif opt.test_subset == 'test':
		subset = 'testing'
	if opt.dataset == 'kinetics':
		test_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'activitynet':
		test_data = ActivityNet(
			opt.video_path,
			opt.annotation_path,
			subset,
			True,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'ucf101':
		test_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'hmdb51':
		test_data = HMDB51(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)

	return test_data
