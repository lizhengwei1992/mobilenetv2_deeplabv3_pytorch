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
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# data
from data import dataset

# model
from model import deeplab_v3_plus

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

# Network 
net = deeplab_v3_plus.DeepLab_v3_plus(nInputChannels=3, n_classes=21)
# if n_gpu > 1:
# 	net = nn.DataParallel(net)

net.to(device)

# Loss 
criterion = nn.CrossEntropyLoss().to(device)

# Data
train_data = dataset.SBD(base_dir=os.path.join(args.dataDir, 'benchmark_RELEASE'), split=['train', 'val'])
test_data  = dataset.VOC(base_dir=os.path.join(args.dataDir, 'VOCdevkit/VOC2012'), split='val')


save = saveData(args)
# finetuning
if args.finetuning:
	net = save.load_model(net)

# Train loop
for epoch in range(args.nEpochs):

	loss_tr = 0

	# optimizer
	lr_ = set_lr(args, epoch)
	optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=0.99, weight_decay=5e-4)
	optimizer.zero_grad() 

	# data load
	trainloader = DataLoader(train_data, batch_size=args.train_batch, 
					drop_last=True, shuffle=True, num_workers=args.nThreads, pin_memory=True)
	testloader = DataLoader(test_data, batch_size=args.test_batch, 
					drop_last=False, shuffle=False, num_workers=args.nThreads, pin_memory=True)


	net.train() 
	print('The {}th epoch ...'.format(epoch))
	for i, sample_batched in enumerate(trainloader):

		optimizer.zero_grad()

		inputs, gts = sample_batched['image'], sample_batched['gt']
		# Forward-Backward of the mini-batch
		inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
		inputs, gts = inputs.to(device), gts.to(device)

		output = net.forward(inputs)				


		n, c, h, w = output.size()
		output = output.contiguous().view(-1, c)
		gts = gts.view(-1)
		loss = criterion(output, gts.long()) / n
		loss_tr = loss.item()

		loss.backward()
		optimizer.step()

	if (epoch+1) % args.save_epoch == 0:

		log = "[{} / {}] \tLearning_rate: {}\t total_loss: {:.4f}".format(epoch+1, 
						args.nEpochs, lr_, loss_tr / args.train_batch)

		print(log)
		save.save_log(log)
		save.save_model(net)




