import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import nltk
import string
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from config import tf_conf

conf = tf_conf()
data_path = '../data/IMDB Dataset processed.csv'
sentences = []

df = pd.read_csv(data_path)
for i in range(len(df)):
    sentences.append(df.iloc[i]['review'])

thresh = 2
total_word = 0
word_count = {}
word_map = {}
word_map_cursor = 0
i = 0
for sentence in sentences:
    if (i+1) % 10000 == 0:
        print(i)
    i += 1
    for word in sentence.split():
        if word in word_count.keys():
            word_count[word] += 1
            if word_count[word] == thresh:
                total_word += 1
                word_map[word] = word_map_cursor
                word_map_cursor += 1
        else:
            word_count[word] = 1

text_features = np.zeros((50000,total_word))
for i in range(len(sentences)):
    if (i+1) % 10000 == 0:
        print(i)
    for word in sentences[i].split():
        if word in word_map.keys():
            text_features[i,word_map[word]] += 1

print("count shape: ",np.sum(text_features,axis=1).shape)
text_features = text_features/np.sum(text_features,axis=1)[:,np.newaxis]
print("text_features shape: ",text_features.shape)
D = 50000
text_features_count = text_features.copy()
text_features_count[text_features_count>0] = 1
text_features_count = np.sum(text_features_count,axis=0)
idf = np.log(D/(text_features_count+1))
print("idf shape: ",idf.shape)
print("idf: ",idf[0:10])
text_features = text_features*idf[np.newaxis,:]
print("text features shape: ",text_features.shape)

svd = TruncatedSVD(1024)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

text_features_1024 = lsa.fit_transform(text_features)
print(text_features_1024.shape)
text_features = np.array(text_features_1024)
np.save("../data/text_features_tfidf_manual_1024.npy",text_features_1024)





