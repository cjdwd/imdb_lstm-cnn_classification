import sys
import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from model.model_tfidf import *
from utils.utils_tfidf import *
from utils.util import *
from config import tfidf_conf, tf_conf
import copy
device = torch.device(0)
torch.cuda.set_device(device)
conf_map = {'tfidf':tfidf_conf(),'tf':tf_conf()}
data_type = sys.argv[1]

'''data wv dataloader setup'''
conf = conf_map[data_type]
data_csv = pd.read_csv(conf.data_path)
text_features = torch.from_numpy(np.load(conf.text_features_path)).cuda()
labels = torch.from_numpy(np.load(conf.labels_path)).cuda()
(train_loader,valid_loader,test_loader) = get_dataloader(data_csv,text_features,labels,conf.batch_size)

'''model loss setup'''
criterion = torch.nn.BCELoss()
model = text_classifier(conf).cuda()
model.load_state_dict(torch.load(conf.save_prefix + conf.name + '_model_max.pth'),strict=True)
model = nn.DataParallel(model)

with torch.set_grad_enabled(False):
    print('Test')
    test_tfidf(test_loader, model)




