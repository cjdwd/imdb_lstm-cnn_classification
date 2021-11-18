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
model.load_state_dict(torch.load(conf.save_prefix + conf.name + '_model_max.pth'),strict=True)
model = nn.DataParallel(model)

with torch.set_grad_enabled(False):
    print('Test')
    test_word2vec(test_loader, model)




