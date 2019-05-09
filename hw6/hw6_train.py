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

train_X = sys.argv[1]
train_Y = sys.argv[2]
dict_path = sys.argv[3]
model_dir = sys.argv[4]
jieba.set_dictionary(dict_path)

rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
# 
def remove_punctuation(line):
    
    line = rule.sub('',line)
    return line
class Preprocess():
    def __init__(self, data_dir, label_dir):
        # Load jieba library 
        
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
        self.t = 0
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
        sentence = remove_punctuation(sentence)
        tokens = list(jieba.cut(sentence,cut_all=False))
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
                    sentence_indices.append(self.word2index[word])
                else:
                    sentence_indices.append(self.word2index[self.unk])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
        print(self.t)
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)        

    def pad_to_len(self, arr, padded_len, padding=0):\
        
        if len(arr) < padded_len:
            for i in range(len(arr),padded_len):
                arr.append(self.word2index[self.pad])
        elif len(arr) > padded_len:
            self.t += 1
            n_arr = []
            for i in range(padded_len):
                n_arr.append(arr[i])
            return n_arr
        return arr
        # TODO
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = Preprocess(train_X, train_Y)
# Get word embedding vectors
embedding = preprocess.get_embedding(load=False)
# Get word indices
data, label = preprocess.get_indices()
# Split train and validation set and create data loader

train_input = data[0:80000]
train_label = label[0:80000]

val_input = data[80000:]
val_label = label[80000:]

def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    lst = np.arange(ylen)
    while (True):
        i = 0
        random.shuffle(lst)
        x = x[lst]
        y = y[lst]
        while(i < loopcount):
#             i = random.randint(0,loopcount)
            i += 1
            yield [x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]]
        
def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def training(train, valid, model, device, mkdir):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    model.train()
    batch_size, n_epoch = 100, 10
    criterion = nn.BCELoss()
#     criterion = nn.CrossEntropyLoss()
    t_batch = 800
    v_batch = 400 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i in range(t_batch):
            data = next(train)
            
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            try: 
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()
                print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                    epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
            except:
                pass
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss/t_batch, total_acc/t_batch*100))

        # validation set
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i in range(v_batch):
                data = next(valid)
                inputs = data[0]
                labels = data[1]
                
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                try:
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)
                    correct = evaluation(outputs, labels)
                    total_acc += (correct / batch_size)
                    total_loss += loss.item()
                except:
                    pass

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/ckpt_{:.3f}".format(model_dir,total_acc/v_batch*100))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        model.train()
        
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
    
train_loader = generate_batch_data_random(train_input, train_label,100)
val_loader = generate_batch_data_random(val_input, val_label,100)

word_dim = 100
hidden_dim = 100
num_layers = 1

model = LSTM_Net(embedding, word_dim, hidden_dim, num_layers)
# model = BILSTM_Net(embedding, word_dim, hidden_dim, num_layers, fix_emb=True)
model = model.to(device)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# Start training
training(train_loader, val_loader, model, device, model_dir)