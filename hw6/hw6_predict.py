import os
import csv
import sys
import argparse
from multiprocessing import Pool

# optional library
import jieba
import pandas as pd
from gensim.models import Word2Vec
import re

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import torch.autograd as autograd

test_path = sys.argv[1]
dict_path = sys.argv[2]
output_path = sys.argv[3]
jieba.set_dictionary(dict_path)

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.3, fix_emb=True):
        super(LSTM_Net, self).__init__()
        # Create embedding layer
#         print(embedding.size(0), embedding.size(1))
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state (maybe we can use more states)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class Preprocess():
    def __init__(self, data_dir, label_dir):
        # Load jieba library 可能要改
#         jieba.load_userdict('dict.txt.big')
        
        self.embed_dim = 100
        self.seq_len = 50
        self.wndw_size = 3
        self.word_cnt = 3
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        # Load corpus
        if data_dir!=None:
            # Read data
            dm = pd.read_csv(data_dir)
            data = dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)
            P.close()
            P.join()
            self.data = data
            
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]

    def tokenize(self, sentence):
        """ Use jieba to tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        # TODO
        sentence = remove_punctuation(sentence)
        tokens = (' '.join(list(jieba.cut(sentence,cut_all=False)))).split(' ')
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=8)
            embed.save(self.save_name)
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1 
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
#             print(len(sentence), sentence)
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                # if word not in word2index append unk index into sentence_indices
                # TODO
                if word in self.word2index:
#                     print("f", self.word2index[word])
                    sentence_indices.append(self.word2index[word])
                else:
                    sentence_indices.append(self.word2index[self.unk])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)        

    def pad_to_len(self, arr, padded_len, padding=0):
        if len(arr) < padded_len:
            for i in range(len(arr),padded_len):
                arr.append(self.word2index[self.pad])
        elif len(arr) > padded_len:
            n_arr = []
            for i in range(padded_len):
                n_arr.append(arr[i])
            return n_arr
        return arr
        # TODO
    def get_embed(self):
        return self.vectors
rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
def remove_punctuation(line):
    
    line = rule.sub('',line)
    return line
preprocess_test = Preprocess(test_path, None)
embedding = preprocess_test.get_embedding(load=True)
# Get word indices
data = preprocess_test.get_indices(test=True)

model = torch.load("final_model")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result = model(data)
output = []
for i in result:
    if i >= 0.6:
        output.append(1)
    else:
        output.append(0)
with open(output_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(output)):
        csv_writer.writerow([i]+[output[i]])