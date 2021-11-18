import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
import re
import string

def custom_preprocessor(text):

    text = text.lower()
    text = re.sub('/<.*?>/', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text

    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''

def sub_symbol_label(dataset):
    symbol = "! @ # $ % ^ & * ( ) _ - + ? / \ = ` ~ \" \' ; : , "
    symbol = symbol.split()
    for i, row in dataset.iterrows():
        review = row['review']
        s = ""
        in_flag = False
        for i in range(len(review)):
            if review[i] == '<':
                s = s + ' '
                in_flag = True
            elif review[i] == '>':
                in_flag = False
                continue

            if in_flag == True:
                continue

            if review[i] in symbol:
                s = s + ' '
            else:
                s = s + review[i]

        review = nltk.word_tokenize(s)
        sentence = []
        full_review = []
        for i in review:
            if i == '.':
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(i.lower())
                full_review.append(i.lower())
        row['review'] = ' '.join(full_review)

data_path = '../data/IMDB Dataset.csv'
sentences = []
df = pd.read_csv(data_path)
df['review']=df['review'].apply(custom_preprocessor)
print(df.iloc[0]['review'])
for i in range(len(df)):
    sentences.append(df.iloc[i]['review'].split())
df.to_csv("../data/IMDB Dataset processed.csv")







