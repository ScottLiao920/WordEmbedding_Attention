#!/usr/bin/env python
# coding: utf-8
import json
import os
import time

import gensim
import nltk
import torch
import torch.nn as nn
from nltk.corpus import gutenberg
from torch.utils.data import Dataset, DataLoader


class PreProcessing:

    def __init__(self):
        self.sents = []
        self.words = []

    def get_sents(self):
        #  Get all sentences in Project Gutenberg
        for file in gutenberg.fileids():
            for sent in gutenberg.sents(file):
                self.sents.append(sent)
        print("Total number of sentences found: {}.\n".format(len(self.sents)))

    def train_gensim(self, path, window, w2v_size, min_count, num_workers):
        # Train a word2vec model using gensim for comparison
        if not self.sents:
            self.get_sents()
        if not os.path.exists(path):
            model = gensim.models.Word2Vec(self.sents, size=w2v_size, window=window, min_count=min_count,
                                           workers=num_workers)
            model.save(path)
        else:
            print("Gensim model already exists!")

    def get_words(self):
        # Create Vocabulary for all the words in Project Gutenberg
        if not self.sents:
            self.get_sents()
        for file in gutenberg.fileids():
            for word in gutenberg.words(file):
                self.words.append(word)
        # words = list(set(words))
        print("Total number of words appeared: {}, including {} unique words.\n".format(len(self.words),
                                                                                        len(list(set(self.words)))))

    def create_vocab(self, path):
        if not self.words:
            self.get_words()
        # write vocabulary sorted according to frequency distribution.
        fd = nltk.FreqDist(self.words)
        vocab = sorted(fd, key=fd.get, reverse=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(vocab, f)
        else:
            print("vocab file already exist")


class w2v_model(nn.Module):
    def __init__(self, settings):
        super(w2v_model, self).__init__()
        self.vocab_size = settings['vocab_size']
        self.batch_size = settings['batch_size']
        self.num_heads = settings['num_heads']
        self.dim_head = settings['dim_head']
        self.num_hidden = self.dim_head * self.num_heads
        self.seq_len = settings['window_size'] * 2
        self.embed_dim = settings['embedding_dim']

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.W_Q = nn.Linear(self.embed_dim, self.num_hidden)
        self.W_K = nn.Linear(self.embed_dim, self.num_hidden)
        self.W_V = nn.Linear(self.embed_dim, self.num_hidden)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def attention(self, target, context):
        Q = self.W_Q(target).view(self.batch_size, self.num_heads, self.dim_head)
        W = torch.zeros([self.batch_size, self.seq_len, self.num_heads, self.num_heads]).to(target.device)
        V = torch.zeros([self.batch_size, self.seq_len, self.num_hidden]).to(target.device)

        for i in range(self.batch_size):
            K_t = self.W_K(target[i]).view(self.num_heads, self.dim_head).transpose(0, 1)
            for j in range(self.seq_len):
                W[i][j] = torch.matmul(Q[i], K_t) / (self.dim_head ** 0.5)
                V[i][j] = self.W_V(target[j])
        W = nn.Softmax(dim=-1)(W)
        V = V.view(self.batch_size, self.seq_len, self.num_heads, self.dim_head)
        tmp = torch.matmul(W, V).view(self.batch_size, self.seq_len, self.num_hidden)
        context_vector = torch.sum(tmp, dim=1).view(self.batch_size, self.num_hidden)
        target_vector = self.W_V(target).view(self.batch_size, self.num_hidden)
        return target_vector, context_vector

    def forward(self, t, c):
        target = self.embedding(t.long())
        context = self.embedding(c.long())
        v_t, v_c = self.attention(target, context)
        return v_t, v_c


class myDataset(Dataset):

    def __init__(self, settings):
        self.window_size = settings['window_size']
        self.dim = settings['embedding_dim']
        # read from project gutenberg
        sents = []
        list(map(sents.extend, list(map(gutenberg.sents, gutenberg.fileids()))))
        print('\n{} sentences fetched.'.format(len(sents)))
        # load vocabulary file
        with open('vocab.json', 'r') as f:
            vocab = json.load(f)
        print('\n{} unique words found in corpus'.format(len(vocab)))
        self.word2id = dict((vocab[i], i) for i in range(len(vocab)))
        self.data = []
        for sent in sents:
            for i in range(len(sent)):
                try:
                    context = [self.word2id[word] for word in sent[max(0, i - self.window_size):i] + sent[i + 1:min(
                        len(sent), i + 1 + self.window_size)]]
                    target = self.word2id[sent[i]]
                    while len(context) < 2 * self.window_size:
                        context.append(0)
                    self.data.append((target, context))
                except KeyError:
                    print(sent[max(0, i - self.window_size):min(len(sent), i + 1 + self.window_size)])
        print('{} pairs found for training'.format(self.__len__()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = torch.Tensor([self.data[index][0]])
        context = torch.Tensor(self.data[index][1])
        return target, context


class pytorch_model(w2v_model, myDataset):

    def __init__(self):
        self.vocab = self.read_vocab('vocab.json')

        self.settings = {
            'vocab_size': len(self.vocab),
            'window_size': 5,
            'num_epochs': 100,
            'embedding_dim': 50,
            'batch_size': 512,
            'num_heads': 12,
            'dim_head': 128,
            'learning_rate': 2e-3
        }
        self.model = w2v_model(settings=self.settings)
        self.lossfunc = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.settings['learning_rate'], momentum=0.9)
        self.dataloader = DataLoader(myDataset(self.settings), batch_size=self.settings['batch_size'], shuffle=True)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def read_vocab(self, path):
        with open(path, 'r') as f:
            tmp = json.load(f)
        return tmp

    def training(self):
        loss_pool = []
        self.model.train()
        start = time.time()
        for epoch in range(self.settings['num_epochs']):
            tmp = []
            for step in range(self.dataset.__len__() // self.settings['batch_size']):
                (t, c) = next(iter(self.dataloader))
                t, c = t.to(self.device), c.to(self.device)
                self.optimizer.zero_grad()
                v_t, v_c = self.model(t, c)
                loss = self.lossfunc(v_t, v_c.to(self.device))
                loss.backward()
                tmp.append(loss.tolist())
                self.optimizer.step()
                if step % 10 == 9:
                    print('epoch {} step {} loss: {:.6f} time used for 10 steps {:6f}'.format(
                        epoch, step, loss.tolist(), time.time() - start))
                    start = time.time()
            torch.save(self.model.state_dict(), 'MSE_SGD/epoch_{}.pt'.format(epoch))
        with open('loss.txt', 'w') as f:
            f.write(str(loss_pool))

    def most_similar(self, token, topk):
