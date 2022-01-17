from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.corr_weights = torch.nn.Parameter(torch.empty(
                1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))
        nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        #f1 = f1.squeeze(1)
        #f2 = f2.squeeze(1)

        #f1_norm = F.normalize(f1, p=2, dim=1, eps=1e-12)
        #f2_norm = F.normalize(f2, p=2, dim=1, eps=1e-12)

        f2_norm_t = f2_norm.transpose(1, 2)

        a1 = torch.matmul(f1_norm, self.corr_weights)
        cc_mat = torch.bmm(a1, f2_norm_t)

        audio_att = F.softmax(cc_mat, dim=1)
        visual_att = F.softmax(cc_mat.transpose(1,2), dim=1)

        atten_audiofeatures = torch.bmm(f1_norm, audio_att)
        atten_visualfeatures = torch.bmm(f2_norm, visual_att)

        return atten_audiofeatures.transpose(1, 2), atten_visualfeatures.transpose(1, 2)
