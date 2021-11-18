import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
import re
import string
from config import word2vec_conf

conf = word2vec_conf()
data_path = '../data/IMDB Dataset processed.csv'
sentences = []
df = pd.read_csv(data_path)
print(df.iloc[0]['review'])
for i in range(len(df)):
    sentences.append(df.iloc[i]['review'].split())

model = Word2Vec(sentences=sentences, vector_size=conf.emb_size, window=5, min_count=1, workers=4)
model.save("../model_pkl/word2vec_"+str(conf.emb_size)+".model")
model = Word2Vec.load("../model_pkl/word2vec_"+str(conf.emb_size)+".model")

y2 = model.wv.most_similar('i', topn=10)
print(model.wv['i'].shape)
for item in y2:
  print(item[0], item[1])


wrong_words = []
text_features = np.zeros((50000, conf.text_size, conf.emb_size))
for j in range(50000):
    row = df.iloc[j]
    review = row['review'].split()
    print(j)
    for i in range(len(review)):
        if i >= conf.text_size:
            break
        try:
            text_features[j,i,:] = model.wv[review[i]]
        except:
            print(review[i])
            if review[i] not in wrong_words:
                wrong_words.append(review[i])
                f = open("../logs/wrong_words.txt", 'a')
                f.write(review[i] + '\n')
                f.close()
np.save('../data/word2vec_text_features.npy',text_features)




