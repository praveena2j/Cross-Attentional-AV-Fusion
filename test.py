from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal

import logging
# import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys

import math
from losses.CCC import CCC



def Test(test_loader, model, criterion, epoch):

	global PublicTest_acc
	global best_PublicTest_acc
	global best_PublicTest_acc_epoch

	model.eval()

	PublicTest_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(test_loader):
		bs, ncrops, c, h, w = np.shape(inputs)
		inputs = inputs.view(-1, c, h, w)
		inputs, targets = inputs.cuda(), targets.cuda()

		targets =  Variable(targets)

		with torch.no_grad():
			inputs = Variable(inputs)

		outputs = model(inputs)
		outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
		loss = criterion(outputs_avg, targets)
		PublicTest_loss += loss.data[0]
		_, predicted = torch.max(outputs_avg.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		#utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		#                   % (PublicTest_loss / (batch_idx + 1), 100.*correct.to(dtype=torch.float)/float(total), correct, total))

	# Save checkpoint.
	PublicTest_acc = 100.*correct.to(dtype=torch.float)/float(total)
	
	if PublicTest_acc > best_PublicTest_acc:
		print('Saving..')
		print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
		state = {
			'net': model.state_dict() ,
			'acc': PublicTest_acc,
			'epoch': epoch,
		}
		if not os.path.isdir(path):
			os.mkdir(path)
		torch.save(state, os.path.join(path,'PublicTest_model.t7'))
		best_PublicTest_acc = PublicTest_acc
		best_PublicTest_acc_epoch = epoch

