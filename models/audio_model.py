import torch
import torch.nn as nn
import sys

class cnn_audio(nn.Module):

	def __init__(self):
		#self.inplanes = 32
		super(cnn_audio, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=[5, 5], stride=(1, 2))
		self.relu1 = nn.ReLU()
		self.bn49 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=[5, 5], stride=(1, 1), padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.bn50 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
		#self.conv3_1 = nn.Conv2d(32, 64, kernel_size=[5, 7], stride=(1, 1), padding=(1, 1))
		#self.bn51 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		#self.relu3_1 = nn.ReLU()
		#self.conv3_2 = nn.Conv2d(384, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		#self.relu3_2 = nn.ReLU()
		#self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		#self.relu3_3 = nn.ReLU()
		#self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)

		self.gru = nn.GRU(4096,128, batch_first=True)

		#self.attention = nn.Sequential(
		#   nn.Linear(self.L, self.D),
		#    nn.Tanh(),
		#    nn.Linear(self.D, self.K)
		#)

		self.rnn = nn.LSTM(
			input_size=1024,
			hidden_size=256,
			num_layers=2,
			batch_first=True,
			#dropout=0.5,
			#bidirectional=True,
			)


		self.fc6 = nn.Conv2d(128, 1024, kernel_size=[14, 1], stride=(1, 1))
		self.bn54 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.relu6 = nn.ReLU()
		self.fc7 = nn.Linear(in_features=1024, out_features=256, bias=True)
		#self.fc7 = nn.Conv2d(128, 256, kernel_size=[1, 4], stride=(1, 1))
		#self.bn55 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bn55 = nn.BatchNorm1d(num_features=256)
		self.dropout = nn.Dropout(0.5)
		self.relu7 = nn.Sigmoid()
		self.fc8 = nn.Linear(in_features=256, out_features=1, bias=True)

	def forward(self, data):
		x1 = self.conv1(data)
		x1_bn = self.bn49(x1)
		x2 = self.relu1(x1_bn)
		x3 = self.pool1(x2)

		x4 = self.conv2(x3)
		x4_bn = self.bn50(x4)
		x5 = self.relu2(x4_bn)
		x6 = self.pool2(x5)

		#x7 = self.conv3_1(x6)
		#print("conv3_1")
		#print(x7.size())
		#x8 = self.relu3_1(x7)
		#x9 = self.conv3_2(x8)
		#x10 = self.relu3_2(x9)
		#x11 = self.conv3_3(x10)
		#x12 = self.relu3_3(x11)
		#x8 = self.pool3(x6)
		#print("pool3")
		#print(x8.size())
		x6_dropout = self.dropout(x6)
		x9 = self.fc6(x6_dropout)
		x9_bn = self.bn54(x9)
		x10_preflatten = self.relu6(x9_bn)

		#x10 = x10_preflatten.view(x10_preflatten.size(0)*x10_preflatten.size(3), -1)
		#print(x10_preflatten.size())
		r_in = x10_preflatten.view(x10_preflatten.shape[0], x10_preflatten.shape[3], -1)   #1,16,4096 batchsize,sequence_length,data_dim
		#r_in = x10.view(1, -1, 1024)   #1,16,4096 batchsize,sequence_length,data_dim

		r_out, (h_n, h_c) = self.rnn(r_in)

		rnn_out = r_out.permute(1, 0, 2)

		#rnn_out = r_out.contiguous().view(r_out.shape[0]*r_out.shape[1], -1)
		#print(rnn_out.size())
		feat = rnn_out[-1]

		#x10_dropout = self.dropout(x10)
		#x11 = self.fc7(x10_dropout)
		#x11_bn = self.bn55(x11)
		#x12 = self.relu7(x11_bn)
		x12_dropout = self.dropout(rnn_out)
		#print(rnn_out.size())
		prediction = self.fc8(x12_dropout)
		#print(prediction.size())
		#sys.exit()
		return feat, prediction
