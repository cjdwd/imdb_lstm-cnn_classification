import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from model.model_word2vec import *
from utils.utils_word2vec import *
from utils.util import *
from config import word2vec_conf
import copy
device = torch.device(0)
torch.cuda.set_device(device)

'''data wv dataloader setup'''
conf = word2vec_conf()
data_csv = pd.read_csv(conf.data_path)
wv = Word2Vec.load(conf.word2vec_path).wv
(train_loader,valid_loader,test_loader) = get_dataloader(data_csv,conf.batch_size)

'''model loss setup'''
criterion = torch.nn.BCELoss()
model = text_classifier().cuda()
model = nn.DataParallel(model)
model_max = copy.deepcopy(model)
max_f1 = 0

for i in range(conf.epochs):
    '''lr decay optimizer setup'''
    if (i+1) % 10 == 0:
        conf.lr = conf.lr / 2
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    '''epoch start'''
    model = model.train()
    total_loss = 0
    y_pred = []
    y_label = []

    '''training'''
    for batch_idx, (text_emb,label) in enumerate(train_loader):
        hidden_state = init_hidden(conf.batch_size,conf.bilstm_layers*2,conf.hidden_size)

        output = model(text_emb.cuda(),hidden_state)
        loss = criterion(output.type(torch.DoubleTensor).squeeze().to(device),
                         label.type(torch.DoubleTensor).squeeze().to(device))

        y_label = y_label + label.detach().cpu().flatten().tolist()
        y_pred = y_pred + output.detach().cpu().flatten().tolist()

        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''print training state'''
        if ((batch_idx+1) % conf.print_step == 0):
            print('Training at Epoch ' + str(i + 1) + ' iteration ' + str(batch_idx+1) + ' with loss ' + str(
              total_loss/(batch_idx+1)))
            #print_metrics(y_pred, y_label)


    '''testing'''
    if (i+1) % conf.test_epo == 0:
        with torch.set_grad_enabled(False):
            print('Valid')
            f1_score = test_word2vec(valid_loader, model)
            if f1_score > max_f1:
                model_max = copy.deepcopy(model)
                max_f1 = f1_score
        print("max f1: ",max_f1)

with torch.set_grad_enabled(False):
    print('Test')
    test_word2vec(test_loader, model_max)

torch.save(model_max.module.state_dict(), conf.save_prefix + conf.name + '_model_max.pth')



