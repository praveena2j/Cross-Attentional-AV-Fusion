#import audio
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
#import VGG_input
import sys
import os
from os import listdir
from scipy import signal
import csv
import librosa
import librosa.display
import torchaudio
import pylab
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from torchvision import transforms

torchaudio.set_audio_backend("sox_io")

def default_audio_reader(videoslist, length, stride):
	sequences = []
	maxVal = 0.711746
	minVal = 0.00 #-0.218993
	for videos in videoslist:
		video_length = len(videos)
		if (video_length < length):
			continue
		images = []
		img_labels = []
		for img in videos:
			imgPath, label = img.strip().split(' ')
			img_labels.append(abs(float(label)))
			#img_labels.append(float(label))
			images.append(imgPath)
		medfiltered_labels = signal.medfilt(img_labels, 3)
		#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
		normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
		normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)*5
		vid = list(zip(images, normalized_labels))
		for i in range(0, video_length-length, stride):
			seq = vid[i : i + length]
			if (len(seq) == length):
				sequences.append(seq)
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		video_length = 0
		videos = []
		lines = list(file)
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

def default_seq_reader(videoslist, data_path):
	label_length = 64
	audio_data = []
	for videos in videoslist:
		video_length = len(videos)
		audio_labels = []
		for img in videos:
			imgPath, label = img.strip().split(' ')
			audio_labels.append(float(label))
		length = len(os.path.dirname(imgPath).split(os.sep))
		file_name = os.path.dirname(imgPath).split(os.sep)[length-1]
		print(os.path.join(data_path, file_name))
		print(os.path.dirname(imgPath))
		medfiltered_labels = signal.medfilt(audio_labels, 3)
		for i in range(0, video_length - label_length, label_length):
			seq = medfiltered_labels[i: i + length]
			audio_data.append(seq)

	return audio_data


def audio_data(file_path, label_path, data_stride, label_stride, timeshift):

	audio_files = os.listdir(file_path)
	audio_file_list = []
	audio_label_list = []
	for audio_file in audio_files:

		with open(label_path + audio_file + '.csv', "r") as f:
			reader = csv.reader(f)
			reader_list = list(reader)
			label_array = np.asarray(reader_list, dtype = np.float32)

			medfiltered_labels = signal.medfilt(label_array[:,0])

			shifted_labels = medfiltered_labels[58:]   ### 68 for arousal , 58 for valence


			num_samples = int(len(shifted_labels)/label_stride)

			for i in range(num_samples):
				audio_filename = os.path.join(file_path, audio_file +"/" + str((i*data_stride)+1) + str('.wav'))

				#print(len(audio_filename))

				####  Extracting Spectrogram
				#y, sr = librosa.load(audio_filename, sr=16000)
				#mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=64, hop_length=40, win_length=400)
				#mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

				#file_name = os.path.splitext(audio_file)[0]
				#if not os.path.exists("audio_spectrograms/" + file):
				#    os.makedirs("audio_spectrograms/" + file)
				#np.save("audio_spectrograms/" + file + "/" + str(i) + ".npy", mel_spect)

				#audio_data = os.path.join("audio_spectrograms", file_name, str(i) + str('.npy'))
				start = i*label_stride
				#audio_file_list.append([audio_data, shifted_labels[i-1]])
				if (len(shifted_labels[start :start+128]) == 128):

					audio_file_list.append(audio_filename)
					audio_label_list.append(shifted_labels[start :start+128])
				#audio_file_list.append([audio_filename, shifted_labels[start :start+150]])
				#print(len(shifted_labels[start :start+150]))
				#audio_file_list.append([mel_spect, shifted_labels[i-1]])
				#vid = list(zip(audio_file_list, audio_label_list))
	return list(zip(audio_file_list, audio_label_list))


class SpecDataset(Dataset):
	def __init__(self, file_path, label_path, data_stride, label_stride, timeshift):
		self.audiolist = audio_data(file_path, label_path, data_stride, label_stride, timeshift)

	def __len__(self):
		return len(self.audiolist)

	def __getitem__(self, idx):
		audiofile, label = self.audiolist[idx]
		spec_image = self.load_data_label(audiofile)
		#spec_image = np.load(spec)
		return spec_image, label


	def load_data_label(self, audioPath):
		#y, sr = librosa.load(audioPath, sr=16000)

		transform_spectra = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
        ])

		#mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=64, hop_length=40, win_length=400)
		#mellog_spc = librosa.power_to_db(mel_spect, ref=np.max)

		#print(mel_spect)

		#audioPath = "../SpeechEmotionRec/audio_trans_wav_new_shift_8_window_64/dev_9/1.wav"
		#print(mel_spect.shape)
		#librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
		#pylab.savefig('new_spectogram.jpg', bbox_inches=None, pad_inches=0)
		#plt.title('Mel Spectrogram');
		#plt.colorbar(format='%+2.0f dB');
		#sys.exit()
		waveform, sr = torchaudio.load(audioPath)

		#if(waveform.shape[1]<96000):

		#print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}\nstd of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean(), waveform.std()))

		#tensor_mean = waveform.mean()
		#tensor_std = waveform.std()
		#norm_waveform = (waveform - tensor_mean) / (tensor_std + 1e-11)

		#tensor_minusmean = waveform - waveform.mean()
		#norm_waveform = tensor_minusmean/tensor_minusmean.abs().max()

		#print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}\nstd of waveform: {}".format(norm_waveform.min(), norm_waveform.max(), norm_waveform.mean(), norm_waveform.std()))
		#sys.exit()

		#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=400, hop_length=40, n_mels=64, n_fft=512, normalized=True)(waveform)
		specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=400, hop_length=79, n_mels=64, n_fft=512, normalized=True)(waveform)

		tensor = specgram.numpy()
		res = np.where(tensor == 0, 1E-19 , tensor)
		spectre = torch.from_numpy(res)

		mellog_spc = spectre.log2()[0,:,:]#.numpy()
		#mel_spect = librosa.amplitude_to_db(mellog_spc, ref=np.max)
		#mel_spect = librosa.amplitude_to_db(mellog_spc, ref=np.max)
		#spec_image = spec_to_image(mellog_spc, eps=1e-6)

		mean = mellog_spc.mean()
		std = mellog_spc.std()
		#mellog_spc_mean = mellog_spc - mean
		#spec_scaled = mellog_spc_mean/mellog_spc_mean.abs().max()

		spec_norm = (mellog_spc - mean) / (std + 1e-11)
		spec_min, spec_max = spec_norm.min(), spec_norm.max()

		#spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)

		spec_scaled = (spec_norm/spec_max)*2 - 1

		#specgram = img_as_ubyte(specgram)
		#specgramImage = transform_spectra(spec_scaled)
		#spec_scaled = spec_scaled.astype(np.uint8)
		#mel_spect = torchaudio.transforms.AmplitudeToDB()(mellog_spc)
		#print(specgram)
		#print(mellog_spc)
		#print(mel_spect)
		#print(type(mel_spect))
		return spec_scaled
