'''
	Train helper functions

Author: Zhengwei Li
Data: July 1 2018
'''
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_lr(args, epoch):

	lrDecay = args.lrDecay
	decayType = args.decayType

	if decayType == 'step':
		epoch_iter = (epoch + 1) // lrDecay
		lr = args.lr / 2**epoch_iter
	elif decayType == 'exp':
		k = math.log(2) / lrDecay
		lr = args.lr * math.exp(-k * epoch)
	elif decayType == 'inv':
		k = 1 / lrDecay
		lr = args.lr / (1 + k * epoch)


	return lr

# Loss 
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class saveData():
	def __init__(self, args):
		self.args = args

		self.save_dir = os.path.join(args.saveDir, args.load)
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.save_dir_model = os.path.join(self.save_dir, 'model')
		if not os.path.exists(self.save_dir_model):
			os.makedirs(self.save_dir_model)

		if os.path.exists(self.save_dir + '/log.txt'):
			self.logFile = open(self.save_dir + '/log.txt', 'a')
		else:
			self.logFile = open(self.save_dir + '/log.txt', 'w')
			
	def save_model(self, model):
	    torch.save(
	        model.state_dict(),
	        self.save_dir_model + '/model_lastest.pt')
	    torch.save(
	        model,
	        self.save_dir_model + '/model_obj.pt')

	def save_log(self, log):
		self.logFile.write(log + '\n')

	def load_model(self, model):
		model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
		print("load mode_status frmo {}/model_lastest.pt".format(self.save_dir_model))
		return model



##############################################
# vis_segmentation
# -------------------
from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap
def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# -------------------------------------------------
# from collections import OrderedDict

# def load_pretrain_pam(net):

# 	state_dict = torch.load('mobilenetv2_718.pth')
# 	new_state_dict = OrderedDict()
# 	for k, v in state_dict.items():
		
# 		if k.startswith('module.feature'):
# 			if not k.startswith('module.features.18'):
# 				# print(k)
# 				name = k[7:] 
# 				new_state_dict[name] = v

# 	net.mobilenet_features.load_state_dict(new_state_dict)

# 	return net

