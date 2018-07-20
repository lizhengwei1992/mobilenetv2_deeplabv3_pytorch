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
import pdb

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

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()


    pdb.set_trace()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask].long()
    loss = F.nll_loss(log_p, target, ignore_index=-1,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


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


##############################################
# load pre_train model 
# -------------------

from collections import OrderedDict
'''
def load_pretrain_pam(net):

	state_dict = torch.load('mobilenetv2_718.pth')
	new_state_dict = OrderedDict()
	i = 0
	for k, v in state_dict.items():
		
		if k.startswith('module.feature'):
			if not k.startswith('module.features.18'):

				print(k)
				name = k[7:] 
				new_state_dict[name] = v

	net.mobilenet_features.load_state_dict(new_state_dict)

	return net
'''
def load_pretrain_pam(net):

	print("Load pre_trained MobileNet_v2 weights from mobilenetv2_718.pth ! ")
	state_dict = torch.load('./pre_train/mobilenetv2_718.pth')
	new_state_dict = OrderedDict()
	i = 0
	n = 0
	for k, v in state_dict.items():
		
		if k.startswith('module.feature'):
			if not k.startswith('module.features.18'):
				# head conv
				if k.split('.')[2] in ['0',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'head_conv') 
					new_state_dict[name] = v

				# block_1
				if k.split('.')[2] in ['1',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_1') 
					new_state_dict[name] = v

				# block_2
				if k.split('.')[2] in ['2',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_2' + '.0') 
					new_state_dict[name] = v

				if k.split('.')[2] in ['3',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_2' + '.1') 
					new_state_dict[name] = v

				# block_3				
				if k.split('.')[2] in ['4',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_3' + '.0') 
					new_state_dict[name] = v
			
				if k.split('.')[2] in ['5',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_3' + '.1') 
					new_state_dict[name] = v

				if k.split('.')[2] in ['6',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_3' + '.2') 
					new_state_dict[name] = v
			
				# block_4				
				if k.split('.')[2] in ['7',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_4' + '.0') 
					new_state_dict[name] = v
					
				if k.split('.')[2] in ['8',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_4' + '.1') 
					new_state_dict[name] = v

				if k.split('.')[2] in ['9',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_4' + '.2') 
					new_state_dict[name] = v
	
				if k.split('.')[2] in ['10',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_4' + '.3') 
					new_state_dict[name] = v
		
				# block_5				
				if k.split('.')[2] in ['11',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_5' + '.0') 
					new_state_dict[name] = v
			
				if k.split('.')[2] in ['12',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_5' + '.1') 
					new_state_dict[name] = v

				if k.split('.')[2] in ['13',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_5' + '.2') 
					new_state_dict[name] = v

				# block_6				
				if k.split('.')[2] in ['14',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_6' + '.0') 
					new_state_dict[name] = v
				
				if k.split('.')[2] in ['15',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_6' + '.1') 
					new_state_dict[name] = v

				if k.split('.')[2] in ['16',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_6' + '.2') 
					new_state_dict[name] = v

				# block_7				
				if k.split('.')[2] in ['17',]:
					name = k.replace((k.split('.')[0]+'.' + k.split('.')[1]+'.' + k.split('.')[2]), 
						'block_7') 
					new_state_dict[name] = v

	net.mobilenet_features.load_state_dict(new_state_dict)

				
	return net

