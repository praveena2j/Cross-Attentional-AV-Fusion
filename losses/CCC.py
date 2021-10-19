import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys

class CCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        prediction = prediction.squeeze()
        ground_truth = ground_truth.squeeze()

        mean_gt = self.mean (ground_truth, 0)
        mean_pred = self.mean (prediction, 0)
        var_gt = self.var (ground_truth, 0)
        var_pred = self.var (prediction, 0)

        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum (v_pred * v_gt) / ((self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2))) + 0.0000001)

        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/(denominator+0.0000001)

        return 1-ccc