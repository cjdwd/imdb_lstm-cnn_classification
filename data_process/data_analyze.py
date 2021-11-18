import sys
sys.path.append("../")

import pandas as pd
import numpy as np
from config import word2vec_conf
conf = word2vec_conf()
dataset = pd.read_csv('../data/IMDB Dataset processed.csv')
print(dataset.iloc[0]['review'])

max_len = 0
avg_len = 0
long_review = 0
for i in range(len(dataset)):
    length = len(dataset.iloc[i]['review'].split())
    max_len = max(max_len,length)
    avg_len = avg_len + length
    if length > conf.text_size:
        long_review += 1

avg_len = avg_len/len(dataset)
print(max_len," ",avg_len)
print('long review: ',long_review)

labels = np.zeros(50000)
for i in range(50000):
    if dataset.iloc[i]['sentiment'] == 'positive':
        labels[i] = 1
    elif dataset.iloc[i]['sentiment'] == 'negative':
        labels[i] = 0
np.save('../data/labels.npy',labels)

