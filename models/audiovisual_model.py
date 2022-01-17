import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
from models.cam import CAM

class cnn_multimodal(nn.Module):

	def __init__(self):
		#self.inplanes = 32
		super(cnn_multimodal, self).__init__()
		#self.visualmodel = visual_model
		#self.audiomodel = audio_model

		self.gru = nn.GRU(4096,128, batch_first=True)
		self.cam = CAM()
		#self.attention = nn.Sequential(
		#   nn.Linear(self.L, self.D),
		#    nn.Tanh(),
		#    nn.Linear(self.D, self.K)
		#)

		self.rnn = nn.LSTM(
			input_size=1024,
			hidden_size=512,
			num_layers=2,
			batch_first=True,
			#dropout=0.5,
			#bidirectional=True,
			)

		self.rnn_av = nn.LSTM(
			input_size=2048,
			hidden_size=512,
			num_layers=2,
			batch_first=True,
			dropout=0.5,
			#bidirectional=True,
			)

		self.fc6 = nn.Conv2d(128, 1024, kernel_size=[5, 5], stride=(1, 1))
		self.bn54 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.relu6 = nn.ReLU()
		self.fc7 = nn.Linear(in_features=256, out_features=128, bias=True)
		#self.fc7 = nn.Conv2d(128, 256, kernel_size=[1, 4], stride=(1, 1))
		#self.bn55 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		#self.bn55 = nn.BatchNorm1d(num_features=128)
		self.dropout = nn.Dropout(0.5)
		#self.relu7 = nn.Sigmoid()
		#self.fc8 = nn.Linear(in_features=128, out_features=1, bias=True)

		self.predictions = nn.Sequential(
				nn.Dropout(0.5),
  				nn.Linear(in_features=512, out_features=256, bias=True),
				#nn.BatchNorm1d(num_features=128),
				nn.ReLU(),
				nn.Linear(in_features=256, out_features=1, bias=True)
		)

		self.modal_predictions = nn.Sequential(
				nn.Dropout(0.5),
  				nn.Linear(in_features=1024, out_features=512, bias=True),
				#nn.BatchNorm1d(num_features=512),
				#nn.ReLU(),
				nn.Tanh(),
				nn.Linear(in_features=512, out_features=256, bias=True),
				#nn.BatchNorm1d(num_features=128),
				#nn.ReLU(),
				nn.Tanh(),
				nn.Linear(in_features=256, out_features=1, bias=True),
    			nn.Tanh()
		)

	def forward(self, audio_feat, visual_feat):

		#audiovisualfeatures = torch.cat((audio_feat, visual_feat), 2)
		#audio_feat = 0.4*audio_feat
		#visual_feat = 0.6*visual_feat
		feat_sum = audio_feat.add(visual_feat)
		#print(tens_add.shape)
		##audiovisualfeatures = torch.mean(tens_add, 1)
		#print(audiovisualfeatures.shape)
		#sys.exit()

		#if (len(features) == 0):
		#	return 0, 0, 0, 0

		#audiovisualfeatures = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
		#seq_lengths, perm_idx = seq_lengths.sort(0, descendingpacked_output=True)

		#visualfeatures = torch.nn.utils.rnn.pad_sequence(vis_features, batch_first=True, padding_value=0)
		#audiofeatures = torch.nn.utils.rnn.pad_sequence(aud_features, batch_first=True, padding_value=0)
		#packed_input = torch.nn.utils.rnn.pack_padded_sequence(audiovisualfeatures, seq_lengths, batch_first=True, enforce_sorted=False)
		#print(packed_input.shape)

		#audfeatures = audiofeatures.view(audiofeatures.shape[0]*audiofeatures.shape[1], -1)
		#vidfeatures = visualfeatures.view(visualfeatures.shape[0]*visualfeatures.shape[1], -1)

		#batch_size, timesteps, feat_dim = audio_feat.size()
		#audio_in = audio_feat.view(batch_size*timesteps, feat_dim)
		#visual_in = visual_feat.view(batch_size*timesteps, feat_dim)

		#print("pool3")
		#print(x8.size())

		#x10_dropout = self.dropout(x10)
		#x11 = self.fc6(x10_dropout)
		#x11_bn = self.bn54(x11)
		#x11_preflatten = self.relu6(x11_bn)

		#x10 = x10_preflatten.view(x10_preflatten.size(0)*x10_preflatten.size(3), -1)
		#print(x10_preflatten.size())
		#r_in = x11_preflatten.view(batch_size, timesteps, -1)   #1,16,4096 batchsize,sequence_length,data_dim

  		#r_in = x10_preflatten.view(x10_preflatten.shape[0], x10_preflatten.shape[3], -1)   #1,16,4096 batchsize,sequence_length,data_dim
		#r_in = x10.view(1, -1, 1024)   #1,16,4096 batchsize,sequence_length,data_dim
		#print(r_in.size())
		#features = features.unsqueeze(0)

		#if (features.shape[2] == 2048):
		#	self.rnn_av.flatten_parameters()
		#	packed_output, (h_n, h_c) = self.rnn_av(features)
		#else:
		#	self.rnn.flatten_parameters()
		#	packed_output, (h_n, h_c) = self.rnn(features)

		#r_out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

		#r_out_visual, (h_n, h_c) = self.rnn(visual_feat)
		#rnn_out = packed_output.view(
		#	packed_output.shape[0]*packed_output.shape[1], -1)

		#r_out = r_out.permute(0, 2, 1)
		#rnn_out = r_out.contiguous().view(r_out.shape[0]*r_out.shape[1], -1)
		#rnn_out_visual = r_out_visual.contiguous().view(r_out_visual.shape[0]*r_out_visual.shape[1], -1)

		#feature = torch.cat((rnn_out_audio, rnn_out_visual), 1)
		out, _ = torch.max(feat_sum,1)
		predictions = self.modal_predictions(out)

		#audio_modal_predictions = self.modal_predictions(audfeatures)
		#visual_modal_predictions = self.modal_predictions(vidfeatures)

		#x10_dropout = self.dropout(rnn_out)
		#x11 = self.fc7(x10_dropout)
		#x11_bn = self.bn55(x11)
		#x12 = self.relu7(x11_bn)
		##x12_dropout = self.dropout(rnn_out)sys.exit()
		##print(rnn_out.size())
		#prediction = self.fc8(x12)
		#refined_targets = torch.stack(refined_targets, 0)


		#refine_targets = torch.nn.utils.rnn.pad_sequence(
		#	refined_targets, batch_first=True, padding_value=0)
		#refine_targets = refine_targets.type(torch.FloatTensor)
		#refine_targets = refine_targets.view(
		#	refine_targets.size(0)*refine_targets.size(1), -1).cuda()
		return predictions#, refine_targets
