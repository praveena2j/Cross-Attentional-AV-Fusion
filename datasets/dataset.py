import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
import torch
from scipy import signal
import bisect
import cv2
import utils.videotransforms as videotransforms
import re
import csv

#def default_seq_reader(videoslist, length, stride):
#		sequences = []
		#maxVal = 0.711746
		#minVal = 0.00 #-0.218993
#		for videos in videoslist:
#				video_length = len(videos)
#				if (video_length < length):
#						continue
#				images = []
#				img_labels = []
#				for img in videos:
#						imgPath, label = img.strip().split(' ')
#						img_labels.append(abs(float(label)))
#						#img_labels.append(float(label))
#						images.append(imgPath)
#				medfiltered_labels = signal.medfilt(img_labels, 3)
#				#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
#				#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)*5
#				vid = list(zip(images, medfiltered_labels))
#				for i in range(0, video_length-length, stride):
#						seq = vid[i : i + length]
#						if (len(seq) == length):
#								sequences.append(seq)
#		#print(len(sequences))
#		return sequences


def default_seq_reader(videoslist, length, stride, audiodata):
	shift_length = length-1
	visualsequences = []
	audiosequences = []
	print(len(videoslist))
	sys.exit()
	for videos in videoslist:
		video_length = len(videos)
		images = []
		img_labels = []
		arr = []
		for img in videos:
			imgPath, label = img.strip().split(' ')
			img_num = int(os.path.splitext(os.path.split(imgPath)[1])[0][5:])
			img_labels.append(float(label))
			images.append(imgPath)
			arr.append(img_num)
		medfiltered_labels = signal.medfilt(img_labels, 3)
		vid = list(zip(images, medfiltered_labels))
		start = 0
		seq_start = 0
		end = start + length
		count = 0
		check_value = shift_length

		while check_value < 7443:  # arr[-1]:
			sub_arr = arr[:end]
			seq_end = bisect.bisect_right(sub_arr, check_value)

			#if (sub_arr[-1] > check_value):
			#	seq_end = i-1
			#else:
			#	seq_end = i
			#print(seq_end)

			if (seq_end > seq_start):
				visualsequences.append([vid[seq_start:seq_end], start])
			else:
				sequences.append([[], start])
			count = count + 1

			if (len(vid[seq_start:seq_end]) > 128):
				print("wrong")
				sys.exit()

			if (start+stride) in arr:
				seq_start = arr.index(start + stride)
			else:
				seq_start = bisect.bisect_left(arr, start+stride)

			#if (len(arr[seq_start:seq_start+128]) == 128):
			#	sequences.append([])
			#if (len(vid[seq_start:seq_end]) == 0):
			#	sequences.append([])
			start = count*stride  # sub_arr[i-1] + 1
			#start = start + stride #sub_arr[i-1] + 1
			end = start + shift_length

			check_value = start + shift_length
	return sequences



def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = []
		lines = list(file)
		#print(len(lines))

		for i in range(9):
			line = lines[video_length]
			#print(line)
			#line = file.readlines()[video_length + i]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			#print(find_str)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			#print(new_video_length)
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

class ImageList(data.Dataset):
	def __init__(self, root, fileList, label_path, length, flag, stride, audiodata, list_reader=default_list_reader, seq_reader=default_seq_reader):

		self.audiodata = audiodata
		self.root = root
		self.label_path = label_path
		self.videoslist = list_reader(fileList)
		self.length = length
		self.stride = stride
		self.sequence_list = seq_reader(self.videoslist, self.length, self.stride, self.audiodata)
		#self.stride = stride
		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader
		self.flag = flag


	def __getitem__(self, index):
		#for video in self.videoslist:
		seq_path, seq_id = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		#seq, label, frame_ids = self.load_data_label(
		seq, label = self.load_data_label(
			self.root, self.label_path, seq_path, self.flag)
		label_index = torch.DoubleTensor([label])
		#else:
		#   seq, label = self.load_test_data_label(seq_path)
		#   label_index = torch.LongTensor([label])
		#if self.transform is not None:
		#    img = self.transform(img)
		return seq, label_index #, seq_id, frame_ids

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, root, label_path, SeqPath, flag):
		#print("Loadung training data")
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
					#transforms.RandomResizedCrop(224),
					#transforms.RandomHorizontalFlip(),
					#transforms.ToTensor(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
				#transforms.Resize(256),
				#transforms.CenterCrop(224),
				#transforms.ToTensor(),
			])
		output = []
		inputs = []
		frame_ids = []
		lab = []
		for image in SeqPath:
			imgPath = image[0]
			head_tail = os.path.normpath(imgPath)
			ind_comps = head_tail.split(os.sep)
			subject_id = ind_comps[-2]
			temp = re.findall(r'\d+', ind_comps[4])
			res = list(map(int, temp))

			with open(label_path + subject_id + '.csv', "r") as f:
				reader = csv.reader(f)
				reader_list = list(reader)
				label_array = np.asarray(reader_list, dtype=np.float32)
				medfiltered_labels = signal.medfilt(label_array[:, 0])
				shifted_labels = medfiltered_labels[58:]  # 68 for arousal , 58 for valence

			label = shifted_labels[res[0]]
			img = cv2.imread(root + imgPath)
			frame_id = os.path.splitext(os.path.split(imgPath)[1])[0][5:]
			#label = image[1]
			#imgPath = image.split(" ")[0]
			#label = image.split(" ")[1]
			#img = Image.open(os.path.join(root, imgPath))
			img = cv2.imread(root + imgPath)
			w,h,c = img.shape
			if w == 0:
				continue
			else:
				img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]
			img = (img/255.)*2 - 1

			#img = img.resize((256,256), Image.ANTIALIAS)
			#inputs.append(data_transforms(img).unsqueeze(0))
			inputs.append(img)
			frame_ids.append(frame_id)
			lab.append(float(label))
			#print(label)
			#label_idx = float(label)
		#label_idx = np.max(lab)
		#print("mean")
		# print(label_idx)
		#output_subset = torch.cat(inputs).unsqueeze(0)
		#output.append(output_subset)
		#print(output_subset.size())
		#print(len(output))
		#print(label_idx)
		imgs=np.asarray(inputs, dtype=np.float32)
		if(imgs.shape[0] != 0):
			imgs = data_transforms(imgs)
			#return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), lab#, frame_ids
			return torch.from_numpy(imgs), lab  # , frame_ids
		#return torch.from_numpy(imgs.transpose([0, 3, 1, 2])), lab, frame_ids
		#return output_subset, lab
		else:
			return [], lab
