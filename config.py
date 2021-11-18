from gensim.models import Word2Vec
import pandas as pd

class word2vec_conf():
    def __init__(self):
        ###model
        self.emb_size = 32
        self.text_size = 512
        self.bilstm_layers = 2
        self.hidden_size = 32
        self.save_prefix = './model_pkl/'
        self.name = 'word2vec'

        ###training
        self.batch_size = 1024
        self.lr = 7e-3
        self.epochs = 50
        self.test_epo = 1
        self.print_step = 4

        ###data and wv
        self.data_path = './data/IMDB Dataset processed.csv'
        self.text_features_path = './data/word2vec_text_features.npy'
        self.labels_path = './data/labels.npy'
        self.word2vec_path = './model_pkl/word2vec_'+str(self.emb_size)+'.model'

class tfidf_conf():
    def __init__(self):
        ###model
        self.text_size = 1024
        self.save_prefix = './model_pkl/'
        self.name = 'tfidf_manual'

        ###training
        self.batch_size = 1024
        self.lr = 7e-3
        self.epochs = 50
        self.test_epo = 1
        self.print_step = 4

        ###data and wv
        self.data_path = './data/IMDB Dataset processed.csv'
        self.text_features_path = './data/text_features_tfidf_manual_1024.npy'
        self.labels_path = './data/labels.npy'

class tf_conf():
    def __init__(self):
        ###model
        self.text_size = 1024
        self.save_prefix = './model_pkl/'
        self.name = 'tf_manual'

        ###training
        self.batch_size = 1024
        self.lr = 7e-3
        self.epochs = 50
        self.test_epo = 1
        self.print_step = 4

        ###data and wv
        self.data_path = './data/IMDB Dataset processed.csv'
        self.text_features_path = './data/text_features_tf_manual_1024.npy'
        self.labels_path = './data/labels.npy'