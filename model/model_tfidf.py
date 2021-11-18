import sys
sys.path.append("../")

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1)
np.random.seed(1)

class text_classifier(torch.nn.Module):

    def __init__(self,conf):
        super(text_classifier,self).__init__()
        self.conf = conf
        self.bn_conv1 = nn.BatchNorm1d(2)
        self.bn_conv2 = nn.BatchNorm1d(4)
        self.bn_conv3 = nn.BatchNorm1d(8)
        self.bn_conv4 = nn.BatchNorm1d(16)
        self.bn_ln1 = nn.BatchNorm1d(32)
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=2,
                               kernel_size=4,stride=2,padding=1)#(n,2,512)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4,
                               kernel_size=4, stride=2, padding=1)#(n,4,256)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=8,
                               kernel_size=4, stride=2, padding=1)#(n,8,128)
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=16,
                               kernel_size=4, stride=2, padding=1)#(n,16,64)

        self.ln1 = nn.Linear(in_features=1024, out_features=32)
        self.ln2 = nn.Linear(in_features=32, out_features=1)

    def forward(self,x):
        x = x.unsqueeze(1)
        # print("after unsqueeze： ",x.shape)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = self.conv4(x)
        x = self.bn_conv4(x)

        x = self.ln1(x.view(x.shape[0], -1))
        x = self.bn_ln1(x)
        x = self.ln2(x)
        # print('after linear: ', x.shape)
        x = F.sigmoid(x)

        return x

