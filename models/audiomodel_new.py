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
		self.pool1 = nn.MaxPool2d(kernel_size=[4, 4], stride=[3, 3], padding=0, dilation=1, ceil_mode=False) # stride=[4, 4],
		self.conv2 = nn.Conv2d(64, 128, kernel_size=[5, 5], stride=(1, 2)) #, padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.conv2_2 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1)) #, padding=(1, 1))
		self.relu2_2 = nn.ReLU()
		self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
		self.conv3_1 = nn.Conv2d(256, 512, kernel_size=[5, 5], stride=(1, 1), padding=(1, 1))
		self.relu3_1 = nn.ReLU()
		#self.bn51 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.conv3_2 = nn.Conv2d(512, 1024, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu3_2 = nn.ReLU()
		self.bn52 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.avg_pool = nn.AvgPool2d(kernel_size=[13, 2],  # [2,7,7]
							   stride=[1, 1])
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
		x5 = self.relu2(x4)
		x6 = self.conv2_2(x5)
		x7 = self.relu2_2(x6)
		x7_bn = self.bn50(x7)
		x8 = self.pool2(x7_bn)
		x9 = self.conv3_1(x8)
		x10 = self.relu3_1(x9)
		x11 = self.conv3_2(x10)
		x12 = self.relu3_2(x11)
		x12_ft = x12.transpose(1, 3).transpose(2,3) #view(16,1,1024,16)
		##x11 = self.conv3_3(x10)
		##x12 = self.relu3_3(x11)
		#x13 = self.avg_pool(x12)

		#x14 = x13.view(x13.size(0)*x13.size(3)*x13.size(2), -1)

		#x14_dropout = self.dropout(x14)
		##x15 = self.fc6(x14_dropout)

		##x9_bn = self.bn54(x9)
		##x10_preflatten = self.relu6(x9_bn)

		##print(x10_preflatten.size())
		##r_in = x10_preflatten.view(x10_preflatten.shape[0]*x10_preflatten.shape[3], -1)   #1,16,4096 batchsize,sequence_length,data_dim
		##r_in = x10.view(1, -1, 1024)   #1,16,4096 batchsize,sequence_length,data_dim
		##print(r_in.size())
		##r_out, (h_n, h_c) = self.rnn(r_in)

		##r_out = r_out.permute(0, 2, 1)
		##rnn_out = r_out.contiguous().view(r_out.shape[0]*r_out.shape[1], -1)
		##print(rnn_out.size())

		##x10_dropout = self.dropout(x10)
		#x11 = self.fc7(x14_dropout)
		#x11_bn = self.bn55(x11)
		#x12 = self.relu7(x11_bn)
		#x12_dropout = self.dropout(x12)
		##print(rnn_out.size())
		#prediction = self.fc8(x12_dropout)
		##print(prediction.size())
		##sys.exit()
		return x12_ft, x12
