import torch
import torch.utils.data as data
import numpy as np
class modelnet(data.Dataset):

	def __init__(self,data):
		self.features = data['features']
		self.class_names = data['targets']

	def __getitem__(self, index):
		shape = self.features[index,].astype(np.float32)
		target = self.class_names[index].astype(np.int64)

		return shape, target

	def __len__(self):
		return len(self.features)
