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
from datasets.audiodataset_new import SpecDataset
#from datasets.audiovisual_dataset import ConcatDataset
from datasets.audiovisualdataset_new import AVList
from models.cam import CAM

#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal

from models.i3d_visual_model import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA
from models.CNN_LSTM import CNN_RNN
from models.audio_model_orig import cnn_audio
from models.audiovisual_model import cnn_multimodal
from models.Vgg_vd_face_fer_dag import Vgg_vd_face_fer_dag
from train_new import train
from val_new import validate
from test import Test
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
#import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
#from fer import FER2013
#from load_imglist import ImageList
from datasets.dataset import ImageList
import math
from losses.CCC_loss import CCCLoss
import wandb

wandb.init(settings=wandb.Settings(start_method="fork"), project='Audio Visual Fusion')
#wandb_logger = WandbLogger(name='BaseFusion', project='AV_Fusion')
#wandb.init(project='Audio Visual Fusion')

parser = argparse.ArgumentParser(description='PyTorch Deep WSDAOR')
parser.add_argument('--arch', '-a', metavar='ARCH', default='WSDA-OR')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers xa(default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
					help='root path of face images (default: none).')
parser.add_argument('-seq_l','--seq-length', default=64, type=int, metavar='N',
					help='sequence length for lstm')
parser.add_argument('-stride','--stride-length', default=64, type=int, metavar='N',
					help='stride length for lstm')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
					help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
					help='path to validation list (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='save root path for features of face images.')
parser.add_argument('--num_classes', default=79077, type=int,
					metavar='N', help='number of classes (default: 79077)')
args = parser.parse_args()

best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_Val_acc = 0  # best PrivateTest accuracy
best_Val_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 60

TrainingAccuracy = []
ValidationAccuracy = []
#def init_weights(m):
#    if type(m) == nn.Linear:
#        torch.nn.init.xavier_uniform(m.weight)
#        m.bias.data.fill_(0.01)

ts = time.time()
Logfile_name = "LogFiles/" + str(ts) + "log_file.log"
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

SEED = 0
### Using seed for deterministic perfromVisual_model_withI3Dg order
if (SEED == 0):
	torch.backends.cudnn.benchmark = True
else:
	print("Using SEED")
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(SEED)

class PadSequence:
	def __call__(self, batch):
		sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
		labels = [x[2] for x in sorted_batch]
		vis_seq_padded = sequences_padded.permute(0,4,1,2,3)
		audio_sequences = torch.stack(aud_sequences)
		return vis_seq_padded, audio_sequences, labels

if not os.path.isdir("SavedWeights"):
	os.makedirs("SavedWeights")

path = "SavedWeights"

print('==> Preparing data..')
label_file = '../../SpeechEmotionRec/ratings_gold_standard/ratings_gold_standard/valence/'
train_data_dir = '../../SpeechEmotionRec/audio_trans_wav_new_shift_1_window_128/train'
dev_data_dir = '../../SpeechEmotionRec/audio_trans_wav_new_shift_1_window_128/dev'

train_data_stride = 1
train_label_stride = 1

dev_data_stride = 1
dev_label_stride = 1
timeshift = 58

AVtraindataset = AVList(root=args.root_path, fileList=args.train_list,
                               label_path=label_file, seq_length=128, subseq_length=16, flag='train', stride=1, audiodata_dir = train_data_dir, timeshift=timeshift)

AVvaldataset = AVList(root=args.root_path, fileList=args.val_list,
                               label_path=label_file, seq_length=128, subseq_length=16, flag='test', stride=1, audiodata_dir = dev_data_dir, timeshift=timeshift)

print(len(AVtraindataset))
print(len(AVvaldataset))
sys.exit()



audiovisualtraindataloader = torch.utils.data.DataLoader(AVtraindataset,
											batch_size=16, shuffle=True,
                                                         num_workers=4, pin_memory=True) #, collate_fn=PadSequence())

audiovisualvaldataloader = torch.utils.data.DataLoader(AVvaldataset,
                                      batch_size=16, shuffle=False,
                                                       num_workers=4, pin_memory=True) #, collate_fn=PadSequence())

print("Size of Audiovisual train data:" +
      str(len(audiovisualtraindataloader)))
print("Size of Audiovisual val data:" +
      str(len(audiovisualvaldataloader)))

print("Data Loaded")
sys.exit()
print("Loading Models")

#visual_model = Vgg_vd_face_fer_dag()
#visual_model = nn.Sequential(*list(visual_model.children())[:-4])
#visual_model = CNN_RNN(visual_model)
i3d = InceptionI3d(400, in_channels=3)
##i3d.load_state_dict(torch.load('PretrainedWeights/rgb_imagenet.pt'))
visual_model = I3D_WSDDA(i3d)
visual_model.cuda()
visual_model = nn.DataParallel(visual_model)
visual_model.load_state_dict(torch.load(
	'PretrainedWeights/valence_cnn_lstm_mil_64_new.t7')['net'])
print(torch.load(
	'PretrainedWeights/valence_cnn_lstm_mil_64_new.t7')['best_Val_acc'])
for param in visual_model.module.i3d_WSDDA.parameters():  # children():
	param.requires_grad = False

audio_model = cnn_audio()
audio_pretrained_model = "PretrainedWeights/valence_audio_cnn_lstm_mil_64_new.t7"
audio_model = nn.DataParallel(audio_model)
audio_model.load_state_dict(torch.load(audio_pretrained_model)['net'])
audio_model.cuda()
print(torch.load(
	'PretrainedWeights/valence_audio_cnn_lstm_mil_64_new.t7')['best_Val_acc'])
for param in audio_model.module.parameters():
	param.requires_grad = False

multimedia_model = cnn_multimodal().cuda()
multimedia_model = nn.DataParallel(multimedia_model)
audiovisual_pretrained_model = "PretrainedWeights/audiovisual_good_model_128_normfeat_concat.t7"
multimedia_model.load_state_dict(torch.load(audiovisual_pretrained_model)['net'])
#print(torch.load(
#	'PretrainedWeights/audiovisual_good_model_128_normfeat_concat.t7')['best_Val_acc'])
#for param in multimedia_model.module.parameters():
#	param.requires_grad = False
#multimedia_model.cuda()
cam = CAM().cuda()
cam.first_init()
print("Models Loaded")

cudnn.benchmark = True
criterion = CCCLoss().cuda()
#optimizer = torch.optim.SGD(list(cam.parameters()) +  list(multimedia_model.parameters()),# filter(lambda p: p.requires_grad, multimedia_model.parameters()),
#								args.lr,
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)

optimizer = torch.optim.Adam(list(cam.parameters()) +  list(multimedia_model.parameters()),# filter(lambda p: p.requires_grad, multimedia_model.parameters()),
								args.lr)
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)

#optimizer = torch.optim.Adam(model.parameters(), lr= 0.001  , amsgrad=True)

for epoch in range(start_epoch, total_epoch):
	#adjust_learning_rate(optimizer, epoch)
	#adjust_learning_rate(optimizer, epoch)

	logging.info("Epoch")
	logging.info(epoch)

	# train for one epoch
	Training_loss, Training_acc = train(
		audiovisualtraindataloader, audio_model, visual_model, multimedia_model, criterion, optimizer, epoch, cam)

	# evaluate on validation set
	Valid_loss, Valid_acc = validate(
		audiovisualvaldataloader, audio_model, visual_model, multimedia_model, criterion, epoch, cam)
	#Test(PrivateTestloader , original_model, criterion, epoch)
	TrainingAccuracy.append(Training_acc)
	ValidationAccuracy.append(Valid_acc)

	logging.info('TrainingAccuracy:')
	logging.info(TrainingAccuracy)

	logging.info('ValidationAccuracy:')
	logging.info(ValidationAccuracy)

	if Valid_acc > best_Val_acc:
		print('Saving..')
		print("best_Val_acc: %0.3f" % Valid_acc)
		state = {
			'net': multimedia_model.state_dict(),
			'best_Val_acc': Valid_acc,
			'best_Val_acc_epoch': epoch,
		}
		if not os.path.isdir(path):
			os.mkdir(path)
		torch.save(state, os.path.join(path,'AV_model_valence_i3d_128_xattention.t7'))
		best_Val_acc = Valid_acc
		best_Val_acc_epoch = epoch

#print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
#print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_Val_acc)
print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
