import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc as AUC
from utils.util import *

torch.manual_seed(1)
np.random.seed(1)

class textDataset(Dataset):
  # Initialize your data, download, etc.
    def __init__(self,data_csv,text_features,labels):
        super(textDataset, self).__init__()
        self.data_csv = data_csv
        self.text_features = text_features
        self.len = len(self.text_features)
        self.labels = labels

    def __getitem__(self, index):
        return self.text_features[index,:].float(), self.labels[index]

    def __len__(self):
        return self.len

def get_dataloader(data_csv,text_features,labels,batch_size):
    train_dataset = textDataset(data_csv=data_csv[0:30000],
                                text_features=text_features[0:30000], labels=labels[0:30000])
    valid_dataset = textDataset(data_csv=data_csv[30000:40000],
                                text_features=text_features[30000:40000], labels=labels[30000:40000])
    test_dataset = textDataset(data_csv=data_csv[40000:50000],
                               text_features=text_features[40000:50000], labels=labels[40000:50000])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return (train_loader,valid_loader,test_loader)

def test_tfidf(data_loader, model):
    model.eval()
    total_loss = 0
    y_pred = []
    y_label = []
    criterion = torch.nn.BCELoss()
    device = torch.device(0)
    for batch_idx, (text_feature,label) in enumerate(data_loader):
        output = model(text_feature)
        loss = criterion(output.type(torch.DoubleTensor).squeeze().to(device),
                         label.type(torch.DoubleTensor).squeeze().to(device))

        y_label = y_label + label.detach().cpu().flatten().tolist()
        y_pred = y_pred + output.detach().cpu().flatten().tolist()
        total_loss += loss.data

    return print_metrics(y_pred,y_label)
