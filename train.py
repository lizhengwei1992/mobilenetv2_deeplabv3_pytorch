'''
	Pytorch implementation of MobileNet_v2_deeplab semantic segmantation  

	Train code 

Author: Zhengwei Li
Data: July 1 2018
'''

import argparse
import timeit
from datetime import datetime														
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# data
from data import dataset

# model
from model import deeplab_v3_plus, deeplab_xception, enet

# dataloder
from data import dataset
# train helper
from utils import *
import pdb

# paramers 
parser = argparse.ArgumentParser()

parser.add_argument('--dataDir', default='./data/', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='save result')
parser.add_argument('--trainData', default='SBD', help='train dataset name')
parser.add_argument('--load', default= 'deeplab_v3_plus', help='save model')

parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
parser.add_argument('--load_pre_train', action='store_true', default=False, help='load pre_train model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='finetuning the training')

parser.add_argument('--nThreads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--train_batch', type=int, default=4, help='input batch size for train')
parser.add_argument('--test_batch', type=int, default=8, help='input batch size for test')
parser.add_argument('--gpus', type=list, default=[0], help='GPUs ID')


parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100)
parser.add_argument('--decayType', default='step')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--save_epoch', type=int, default=5, help='number of epochs to save model')


args = parser.parse_args()



# Multi-GPUs
if args.without_gpu:
	print("use CPU !")
	device = torch.device('cpu')
else:
	if torch.cuda.is_available():
		n_gpu = torch.cuda.device_count()
		print("----------------------------------------------------------")
		print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
		print("----------------------------------------------------------")

		device = torch.device('cuda')
#-----------------------------------------------------
# Network 
#---------------

# net = deeplab_v3_plus.DeepLabv_v3_plus_mv2_os_32(nInputChannels=3, n_classes=1)
net = deeplab_v3_plus.DeepLabv_v3_plus_mv2_os_8(nInputChannels=3, n_classes=1)

# net = enet.ENet(5)
# net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=15) 

# if n_gpu > 1:
# 	net = nn.DataParallel(net)

if args.load_pre_train:
	net = load_pretrain_pam(net)

net.to(device)

#-----------------------------------------------------
# Loss 
#---------------
# criterion = nn.CrossEntropyLoss(weight=None, size_average=False, ignore_index=-1).to(device)
criterion = nn.BCELoss()

#-----------------------------------------------------
# Data
#---------------

# train_data = dataset.SBD(base_dir=os.path.join(args.dataDir, 'benchmark_RELEASE'), split=['train', 'val'])
# test_data  = dataset.VOC(base_dir=os.path.join(args.dataDir, 'VOCdevkit/VOC2012'), split='val')
train_data = getattr(dataset, args.trainData)(base_dir = args.dataDir)

# data load
trainloader = DataLoader(train_data, batch_size=args.train_batch, 
				drop_last=True, shuffle=True, num_workers=args.nThreads, pin_memory=False)

#-----------------------------------------------------
# Train loop
#---------------

save = saveData(args)
# finetuning
if args.finetuning:
	net = save.load_model(net)

print("Start Train ! ... ...")
for epoch in range(args.nEpochs):

	loss_tr = 0
	loss_ = 0

	# optimizer
	# for param in net.mobilenet_features.parameters():
	# 	param.requires_grad = False

	lr_ = set_lr(args, epoch)
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr_, momentum=0.99, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

	net.train() 

	for i, sample_batched in enumerate(trainloader):

		inputs, gts = sample_batched['image'], sample_batched['gt']

		inputs, gts = inputs.to(device), gts.to(device)

		output = net.forward(inputs)				

		output = F.sigmoid(output)
		output_flat = output.view(-1)
		gts = gts.view(-1)
		loss = criterion(output_flat, gts)


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_ += loss.item()

	loss_tr = loss_ / (i+1)
	if (epoch+1) % args.save_epoch == 0:

		log = "[{} / {}] \tLearning_rate: {}\t total_loss: {:.5f}".format(epoch+1, 
						args.nEpochs, lr_, loss_tr)

		print(log)
		save.save_log(log)
		save.save_model(net)




