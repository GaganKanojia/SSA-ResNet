import torch
from torch import nn

from models import SSAresnet, SSAwide_resnet, SSAresnext, SSAresnet_modelnet
import numpy as np

def generate_model(opt):
	assert opt.model in [
		'SSAresnet', 'SSAwideresnet', 'SSAresnext','SSAresnet_modelnet']

	if opt.model == 'SSAresnet':
		assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

		from models.SSAresnet import get_fine_tuning_parameters

		if opt.model_depth == 10:
			model = SSAresnet.SSAresnet10(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 18:
			model = SSAresnet.SSAresnet18(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 34:
			model = SSAresnet.SSAresnet34(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 50:
			model = SSAresnet.SSAresnet50(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 101:
			model = SSAresnet.SSAresnet101(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 152:
			model = SSAresnet.SSAresnet152(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 200:
			model = SSAresnet.SSAresnet200(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
	elif opt.model == 'SSAwideresnet':
		assert opt.model_depth in [50]

		from models.SSAwide_resnet import get_fine_tuning_parameters

		if opt.model_depth == 50:
			model = SSAwide_resnet.SSAresnet50(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				k=opt.wide_resnet_k,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
	elif opt.model == 'SSAresnext':
		assert opt.model_depth in [50, 101, 152]

		from models.SSAresnext import get_fine_tuning_parameters

		if opt.model_depth == 50:
			model = SSAresnext.SSAresnet50(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				cardinality=opt.resnext_cardinality,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 101:
			model = SSAresnext.SSAresnet101(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				cardinality=opt.resnext_cardinality,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
		elif opt.model_depth == 152:
			model = SSAresnext.SSAresnet152(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				cardinality=opt.resnext_cardinality,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)

	elif opt.model == 'SSAresnet_modelnet':
		assert opt.model_depth in [8]

		from models.SSAresnet_modelnet import get_fine_tuning_parameters

		if opt.model_depth == 8:
			model = SSAresnet_modelnet.SSAresnet8(
				num_classes=opt.n_classes,
				shortcut_type=opt.resnet_shortcut,
				sample_size=opt.sample_size,
				sample_duration=opt.sample_duration)
	

	if not opt.no_cuda:

		################################################# Number of parameters in the model
		## Reference- https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
		total_params = 0
		net = model
		for x in filter(lambda p: p.requires_grad, net.parameters()):
			total_params += np.prod(x.data.numpy().shape)
		print("Total number of params", total_params)
		print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
		################################################################################################

		model = model.cuda()
		model = nn.DataParallel(model, device_ids=None)

		if opt.pretrain_path:
			print('loading pretrained model {}'.format(opt.pretrain_path))
			pretrain = torch.load(opt.pretrain_path)
			assert opt.arch == pretrain['arch']

			model.load_state_dict(pretrain['state_dict'])

			model.module.fc = nn.Linear(model.module.fc.in_features,
										opt.n_finetune_classes)
			model.module.fc = model.module.fc.cuda()

			parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
			return model, parameters
	else:
		if opt.pretrain_path:
			print('loading pretrained model {}'.format(opt.pretrain_path))
			pretrain = torch.load(opt.pretrain_path)
			assert opt.arch == pretrain['arch']

			model.load_state_dict(pretrain['state_dict'])

			model.fc = nn.Linear(model.fc.in_features,opt.n_finetune_classes)

			parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
			return model, parameters

	return model, model.parameters()
