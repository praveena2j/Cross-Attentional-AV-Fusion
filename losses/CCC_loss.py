import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys

class CCCLoss(nn.Module):
	def __init__(self):
		super().__init__()



	def forward(self, x, y):
		x = x.squeeze()
		y = y.squeeze()
		x_mean = torch.mean(x)
		y_mean = torch.mean(y)
		vx = x - x_mean
		vy = y - y_mean
		rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + 1e-08)
		x_s = torch.std(x)
		y_s = torch.std(y)
		ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_mean - y_mean, 2) + 1e-08)
		return 1-ccc