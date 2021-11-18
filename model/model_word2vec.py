import sys
sys.path.append("../")

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from config import word2vec_conf
torch.manual_seed(1)
np.random.seed(1)

class text_classifier(torch.nn.Module):

    def __init__(self):
        super(text_classifier,self).__init__()
        self.conf = word2vec_conf()
        # self.rnn = nn.RNN(input_size=self.conf.emb_size,hidden_size=self.conf.hidden_size,num_layers=6)
        self.lstm = torch.nn.LSTM(input_size=self.conf.emb_size, hidden_size=self.conf.hidden_size,
                                  num_layers=self.conf.bilstm_layers, batch_first=True, bidirectional=True,
                                  dropout=0.1)
        self.bn1 = nn.BatchNorm1d(self.conf.text_size)
        self.bn2 = nn.BatchNorm1d(self.conf.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.conf.hidden_size//2)
        self.bn4 = nn.BatchNorm1d(self.conf.hidden_size//4)
        self.conv1 = nn.Conv1d(in_channels=2*self.conf.hidden_size,out_channels=self.conf.hidden_size,
                               kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.conf.hidden_size, out_channels=self.conf.hidden_size//2,
                               kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.conf.hidden_size//2, out_channels=self.conf.hidden_size//4,
                               kernel_size=4, stride=2, padding=1)
        self.linear = nn.Linear(in_features=(self.conf.text_size//8)*self.conf.hidden_size//4, out_features=1)

    def forward(self,x,hidden_state):
        # print('review_emb: ',x.shape)
        # x = self.rnn(x)[0]
        x = self.lstm(x,hidden_state)[0]
        # print('after rnn: ', x.shape)

        x = self.bn1(x)
        #print('after bn: ', x.shape)

        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.linear(x.view(x.shape[0], -1))
        # print('after linear: ', x.shape)

        x = F.sigmoid(x)
        # print('after sigmoid: ', x.shape)

        return x

