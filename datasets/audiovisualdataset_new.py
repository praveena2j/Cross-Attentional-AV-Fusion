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
import librosa
import librosa.display
import torchaudio
from scipy import signal
import bisect
import cv2
import utils.videotransforms as videotransforms
import re
import csv
import time

torchaudio.set_audio_backend("sox_io")

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

def audio_data(file_path, label_path, stride, timeshift):

	audio_files = os.listdir(file_path)
	audio_file_list = []
	audio_label_list = []
	audio_labels_list = {}
	audio_files_list = {}

	for audio_file in audio_files:
		audio_file_list = []
		audio_label_list = []
		with open(label_path + audio_file + '.csv', "r") as f:
			reader = csv.reader(f)
			reader_list = list(reader)
			label_array = np.asarray(reader_list, dtype = np.float32)
			medfiltered_labels = signal.medfilt(label_array[:,0])

			shifted_labels = medfiltered_labels[timeshift:]   ### 68 for arousal , 58 for valence
			#num_samples = 7443 #len(reader_list)
			#print(num_samples = 7443
			num_samples = int(len(shifted_labels)/stride)

		for i in range(num_samples):
			audio_filename = os.path.join(file_path, audio_file +"/" + str((i*stride)+1) + str('.wav'))
			start = i*stride
			if (len(shifted_labels[start :start+128]) == 128):
				audio_file_list.append(audio_filename)
				label = torch.FloatTensor([shifted_labels[start :start+128]])
				audio_label_list.append(torch.mean(label, dim=1))
				#audio_file_list.append([audio_filename, shifted_labels[start :start+150]])
				#print(len(shifted_labels[start :start+150]))
				#audio_file_list.append([mel_spect, shifted_labels[i-1]])
				#vid = list(zip(audio_file_list, audio_label_list))
		#audio_files_list.append([{audio_file : audio_file_list}])

		audio_files_list[audio_file] = audio_file_list
		audio_labels_list[audio_file] = audio_label_list
		#audio_labels_list.append(audio_label_list)
	return audio_files_list, audio_labels_list #list(zip(audio_files_list, audio_labels_list))

def default_seq_reader(visualdatalist, seq_length, subseq_length, stride, flag, audiodatalist, labels):
	shift_length = seq_length-1
	sequences = []
	audiosequences = []
	if flag == 'train':
		print("train mode")
		subs = ['train_1', 'train_2','train_3','train_4','train_5','train_6','train_7','train_8','train_9']
	else:
		print("dev mode")
		subs = ['dev_1', 'dev_2','dev_3','dev_4','dev_5','dev_6','dev_7','dev_8','dev_9']
	for sub in subs:
		visdata = visualdatalist[sub]
		auddata = audiodatalist[sub]
		labdata = labels[sub]
		images = []
		img_labels = []
		arr = []
		for img in visdata:
			imgPath, label = img.strip().split(' ')
			img_num = int(os.path.splitext(os.path.split(imgPath)[1])[0][5:])
			#img_labels.append(float(label))
			images.append(imgPath)
			arr.append(img_num)
		#medfiltered_labels = signal.medfilt(img_labels, 3)
		#vid = list(zip(images, medfiltered_labels))
		start = 0
		seq_start = 0
		end = start + seq_length
		count = 0
		check_value = shift_length
		while check_value < 7443:  # arr[-1]:
			sub_arr = arr[:end]
			seq_end = bisect.bisect_right(sub_arr, check_value) #-1
			if (seq_end > seq_start):
				if (len(images[seq_start:seq_end]) > 127):
				#if (len(vid[seq_start:seq_end]) > 32):
					#visualsequences.append([vid[seq_start:seq_end], start])
					#audiosequences.append(auddata[count])
					sequences.append([images[seq_start:seq_end], auddata[count], labdata[count]])
			count = count + 1

			if (start+stride) in arr:
				seq_start = arr.index(start + stride)
			else:
				seq_start = bisect.bisect_left(arr, start+stride)
			start = count*stride  # sub_arr[i-1] + 1
			#start = start + stride #sub_arr[i-1] + 1
			end = start + shift_length
			check_value = start + shift_length
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = {}
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
					sub = os.path.split(find_str)[1]
					new_video_length = new_video_length + 1
			#print(new_video_length)
			videos[sub] = lines[video_length:video_length + new_video_length]
			#videos.append([{sub : lines[video_length:video_length + new_video_length]}])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

class AVList(data.Dataset):
	def __init__(self, root, fileList, label_path, seq_length, subseq_length, flag, stride, audiodata_dir, timeshift, list_reader=default_list_reader, seq_reader=default_seq_reader):

		self.audiodatalist, self.labels = audio_data(audiodata_dir, label_path, stride, timeshift)
		self.root = root
		self.label_path = label_path
		self.visualdatalist = list_reader(fileList)
		self.seq_length = seq_length
		self.subseq_length = subseq_length
		self.num_subseqs = self.seq_length / self.subseq_length
		self.stride = stride
		self.flag = flag
		self.sequence_list = seq_reader(self.visualdatalist, self.seq_length, self.subseq_length, self.stride, self.flag, self.audiodatalist, self.labels)
		#self.stride = stride
		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader



	def __getitem__(self, index):

		#for video in self.videoslist:
		visualsequence, audiosequence, lab = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		#seq, label, frame_ids = self.load_data_label(
		#torch.cuda.synchronize()
		#t11 = time.time()
		visualseq = self.load_visualdata_label(
			self.root, self.label_path, visualsequence, self.flag, self.subseq_length)
		#torch.cuda.synchronize()
		#t12 = time.time()
		#print('visual data loading time', t12-t11)

		#torch.cuda.synchronize()
		#t13 = time.time()
		audioseq = self.load_audiodata_label(audiosequence, self.num_subseqs)
		#torch.cuda.synchronize()
		#t14 = time.time()
		#print('audio data loading time', t14-t13)
		return visualseq, audioseq, lab #, seq_id, frame_ids

	def __len__(self):
		return len(self.sequence_list)

	def load_visualdata_label(self, root, label_path, SeqPath, flag, subseq_len):
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
		subseq_inputs = []
		frame_ids = []
		lab = []
		count_subseqs = 0
		for imgPath in SeqPath:
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
			if len(inputs) == subseq_len:
				subseq_inputs.append(inputs)
				inputs = []
			#label_idx = float(label)
		#label_idx = np.max(lab)
		#print("mean")
		# print(label_idx)
		#output_subset = torch.cat(inputs).unsqueeze(0)
		#output.append(output_subset)
		#print(output_subset.size())
		#print(len(output))
		#print(label_idx)
		seqs = []
		for subseq in subseq_inputs:
			imgs=np.asarray(subseq, dtype=np.float32)
			#if(imgs.shape[0] != 0):
			imgs = data_transforms(imgs)
			seqs.append(torch.from_numpy(imgs))
		vid_seqs = torch.stack(seqs).permute(4,0,1,2,3)
		#return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), lab#, frame_ids
		return vid_seqs #torch.from_numpy(imgs) #, lab  # , frame_ids
		#return torch.from_numpy(imgs.transpose([0, 3, 1, 2])), lab, frame_ids
		#return output_subset, lab
		#else:
		#	return []


	def load_audiodata_label(self, audioPath, num_subseqs):
		#y, sr = librosa.load(audioPath, sr=16000)
		#torch.cuda.synchronize()
		#t11 = time.time()
		transform_spectra = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
        ])

		waveform, sr = torchaudio.load(audioPath)
		subseq_len = waveform.shape[1] / num_subseqs
		spectrograms = []
		for i in range(int(num_subseqs)):
			waveform, sr = torchaudio.load(audioPath, frame_offset=int(subseq_len*i), num_frames=int(subseq_len))

			#if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
			#	waveform, sr = torchaudio.load(audioPath)
			#else:
			#	waveform, sr = torchaudio.load(audioPath)
			specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=400, hop_length=160, n_mels=128, n_fft=1024, normalized=True)(waveform)
			#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=640, hop_length=640, n_mels=128, n_fft=1024, normalized=True)(waveform)
			tensor = specgram.numpy()
			res = np.where(tensor == 0, 1E-19 , tensor)
			spectre = torch.from_numpy(res)

			mellog_spc = spectre.log2()[0,:,:]#.numpy()
			mean = mellog_spc.mean()
			std = mellog_spc.std()
			spec_norm = (mellog_spc - mean) / (std + 1e-11)
			spec_min, spec_max = spec_norm.min(), spec_norm.max()
			spec_scaled = (spec_norm/spec_max)*2 - 1
			spectrograms.append(spec_scaled)
		melspecs_scaled = torch.stack(spectrograms)
		#torch.cuda.synchronize()
		#t12 = time.time()
		return melspecs_scaled