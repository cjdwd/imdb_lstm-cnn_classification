import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc as AUC

torch.manual_seed(1)
np.random.seed(1)

def init_hidden(batch_size, num_gpus, input_size):
  return (torch.zeros(int(4*num_gpus), int(batch_size // num_gpus), input_size),
          torch.zeros(int(4*num_gpus), int(batch_size // num_gpus), input_size))

def print_metrics(y_pred,y_label):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    precision = precision_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred)
    f1 = 2*recall*precision/(recall+precision+1e-6)
    print('Precision: ',precision)
    print('Recall: ',recall)
    print('F1: ',f1)
    return  f1