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
from tensorboardX import SummaryWriter
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
            for j in range(self.seq_len):
                K_t = self.W_K(context[i][j]).view(self.num_heads, self.dim_head).transpose(0, 1)
                W[i][j] = torch.matmul(Q[i], K_t) / (self.dim_head ** 0.5)
                V[i][j] = self.W_V(context[i][j])
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


class w2v_model_CBoW(nn.Module):
    def __init__(self, settings):
        super(w2v_model_CBoW, self).__init__()
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
        self.W_out = nn.Linear(self.num_hidden, self.vocab_size)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def attention(self, target, context):
        Q = self.W_Q(target).view(self.batch_size, self.num_heads, self.dim_head)
        W = torch.zeros([self.batch_size, self.seq_len, self.num_heads, self.num_heads]).to(target.device)
        V = torch.zeros([self.batch_size, self.seq_len, self.num_hidden]).to(target.device)

        for i in range(self.batch_size):
            for j in range(self.seq_len):
                K_t = self.W_K(context[i][j]).view(self.num_heads, self.dim_head).transpose(0, 1)
                W[i][j] = torch.matmul(Q[i], K_t) / (self.dim_head ** 0.5)
                V[i][j] = self.W_V(context[i][j])
        W = nn.Softmax(dim=-1)(W)
        V = V.view(self.batch_size, self.seq_len, self.num_heads, self.dim_head)
        tmp = torch.matmul(W, V).view(self.batch_size, self.seq_len, self.num_hidden)
        context_vector = torch.sum(tmp, dim=1).view(self.batch_size, self.num_hidden)
        return context_vector

    def forward(self, t, c):
        target = self.embedding(t.long())
        context = self.embedding(c.long())
        v_c = self.attention(target, context)
        pred = nn.Softmax(dim=1)(self.W_out(v_c))
        return pred


class pytorch_model(w2v_model, w2v_model_CBoW, myDataset):
    def __init__(self, mode, settings):
        self.vocab = self.read_vocab('vocab.json')
        if not settings:
            self.settings = {
                'vocab_size': len(self.vocab),
                'window_size': 5,
                'num_epochs': 3,
                'embedding_dim': 50,
                'batch_size': 512,
                'num_heads': 12,
                'dim_head': 128,
                'learning_rate': 2e-3
            }
        else:
            self.settings = settings
        print(self.settings)
        super(pytorch_model, self).__init__(self.settings)

        # create model object
        if mode == 'MSE':
            self.model = w2v_model(settings=self.settings)
            self.lossfunc = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.settings['learning_rate'], momentum=0.9)
        elif mode == 'COS':
            self.model = w2v_model(settings=self.settings)
            self.lossfunc = nn.CosineEmbeddingLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.settings['learning_rate'], momentum=0.9)
        elif mode == 'CBoW':
            self.model = w2v_model_CBoW(settings=self.settings)
            self.lossfunc = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings['learning_rate'])
        else:
            raise Exception('mode must be one of CBoW, MSE, COS!')

        # create dataloader
        dataset = myDataset(self.settings)
        uni_leng = dataset.__len__() // 10
        ttl_leng = dataset.__len__()
        train_set, test_set, dev_set = torch.utils.data.random_split(dataset,
                                                                     [uni_leng * 8, uni_leng, ttl_leng - 9 * uni_leng])

        self.train_loader = DataLoader(train_set, batch_size=self.settings['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.settings['batch_size'], shuffle=True)
        self.dev_loader = DataLoader(dev_set, batch_size=self.settings['batch_size'], shuffle=True)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mode = mode
        self.writer = SummaryWriter('logs/' + mode)

    def read_vocab(self, path):
        with open(path, 'r') as f:
            tmp = json.load(f)
        return tmp

    def forward_pass(self, t, c):
        if self.mode != 'CBoW':
            v_t, v_c = self.model(t, c)
            loss = self.lossfunc(v_t, v_c.to(self.device))
        else:
            pred = self.model(t, c)
            loss = self.lossfunc(pred, t.long().view(-1))
        return loss

    def train(self):
        self.model.train()
        num_steps = self.train_loader.dataset.__len__() // self.settings['batch_size']
        for epoch in range(self.settings['num_epochs']):
            start = time.time()
            for step in range(num_steps):
                (t, c) = next(iter(self.train_loader))
                t, c = t.to(self.device), c.to(self.device)
                self.optimizer.zero_grad()
                loss = self.forward_pass(t, c)
                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    print('epoch {} step {} loss: {:.6f}, time used for 100 steps: {:6f} seconds'.format(
                        epoch, step, loss.tolist(), time.time() - start))
                    (t, c) = next(iter(self.test_loader))
                    test_loss = self.forward_pass(t, c)
                    (t, c) = next(iter(self.dev_loader))
                    dev_loss = self.forward_pass(t, c)
                    self.writer.add_scalars('loss', {
                        'train': loss.tolist(),
                        'test': test_loss.tolist(),
                        'dev': dev_loss.tolist()
                    }, epoch * num_steps + step)
                    start = time.time()
            torch.save(self.model.state_dict(), 'MSE_SGD/epoch_{}.pt'.format(epoch))
        print("Done training! Writing embedding into directory.")

    def get_embed(self, token):
        return self.model.embedding(torch.Tensor([self.vocab.index(token)]).long().to(self.device))

    def most_similar(self, token, topk):
        v_w1 = self.get_embed(token)
        word_sim = {}
        for i in range(len(self.vocab)):
            word = self.vocab[i]
            v_w2 = self.get_embed(word)
            theta = self.cos_sim(v_w1, v_w2)
            word_sim[word] = theta.detach().numpy()
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        for word, sim in words_sorted[:topk]:
            yield (word, sim)
