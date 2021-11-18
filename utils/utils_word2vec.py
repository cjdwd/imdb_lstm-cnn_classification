import sys
sys.path.append("../")

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc as AUC
from config import word2vec_conf
from utils.util import *

torch.manual_seed(1)
np.random.seed(1)

conf = word2vec_conf()

class textDataset(Dataset):
  # Initialize your data, download, etc.
    def __init__(self, data_csv, text_features, labels):
        super(textDataset, self).__init__()
        self.data_csv = data_csv
        self.len = len(data_csv)
        self.text_features = text_features
        self.labels = labels
        self.wrong_words = []

    def __getitem__(self, index):
        review_emb = self.text_features[index,:,:]
        label = self.labels[index]
        return review_emb.float(), label

    def __len__(self):
        return self.len

def get_dataloader(data_csv,batch_size):
    text_features = torch.from_numpy(np.load(conf.text_features_path)).cuda()
    labels = torch.from_numpy(np.load(conf.labels_path)).cuda()
    train_dataset = textDataset(data_csv=data_csv[0:30000],
                                text_features=text_features[0:30000],labels=labels[0:30000])
    valid_dataset = textDataset(data_csv=data_csv[30000:40000],
                                text_features=text_features[30000:40000],labels=labels[30000:40000])
    test_dataset = textDataset(data_csv=data_csv[40000:50000],
                               text_features=text_features[40000:50000],labels=labels[40000:50000])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return (train_loader,valid_loader,test_loader)

def test_word2vec(data_loader, model):
    model.eval()
    total_loss = 0
    y_pred = []
    y_label = []
    criterion = torch.nn.BCELoss()
    device = torch.device(0)
    for batch_idx, (text_emb,label) in enumerate(data_loader):
        hidden_state = init_hidden(conf.batch_size,conf.bilstm_layers*2,conf.emb_size)

        output = model(text_emb.cuda(),hidden_state)
        loss = criterion(output.type(torch.DoubleTensor).squeeze().to(device),
                         label.type(torch.DoubleTensor).squeeze().to(device))
        y_label = y_label + label.flatten().tolist()
        y_pred = y_pred + output.flatten().tolist()
        total_loss += loss.data

    return print_metrics(y_pred,y_label)
