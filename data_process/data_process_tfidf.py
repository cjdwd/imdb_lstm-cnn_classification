import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD


data_path = '../data/IMDB Dataset processed.csv'
sentences = []

df = pd.read_csv(data_path)
for i in range(len(df)):
    sentences.append(df.iloc[i]['review'])

tfv = TfidfVectorizer(min_df=1,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True,
                      stop_words='english')
tfv.fit(sentences)
text_features = tfv.transform(sentences)
print(text_features.shape)
print(tfv.get_feature_names()[0:100])

svd = TruncatedSVD(1024)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

text_features_1024 = lsa.fit_transform(text_features)
print(text_features_1024.shape)
text_features = np.array(text_features_1024)
np.save("../data/text_features_1024.npy",text_features_1024)





