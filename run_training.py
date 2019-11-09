#!/usr/bin/env python
# coding: utf-8
import json
import os
import time

import torch
import torch.nn as nn
from nltk.corpus import gutenberg
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

os.environ['cuda_visible_device'] = '1'

torch.manual_seed(0)

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

settings = {
    'vocab_size': len(vocab),
    'window_size': 5,
    'num_epochs': 100,
    'embedding_dim': 50,
    'batch_size': 512,
    'num_heads': 12,
    'dim_head': 128,
    'learning_rate': 2e-3
}


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


dataset = myDataset(settings)
uni_leng = dataset.__len__() // 10
leng = dataset.__len__()
train_set, test_set, dev_set = torch.utils.data.random_split(dataset, [uni_leng*8, uni_leng, leng-9*uni_leng])

train_loader = DataLoader(train_set, batch_size=settings['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=settings['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=settings['batch_size'], shuffle=True)

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


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)

model = w2v_model(settings).to(device)
lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'], momentum=0.9)
writer = SummaryWriter()


model.train()
num_steps = train_set.__len__()//settings['batch_size']
for epoch in range(settings['num_epochs']):
    for step in range(train_set.__len__()//settings['batch_size']):
        start = time.time()
        (t, c) = next(iter(train_loader))
        t, c = t.to(device), c.to(device)
        optimizer.zero_grad()
        v_t, v_c = model(t, c)
        loss = lossfunc(v_t, v_c.to(device))
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print('epoch {} step {} loss: {:.6f} time used for 10 steps {:6f}'.format(
                epoch, step, loss.tolist(), time.time()-start))
            writer.add_scalar('speed', time.time()-start, epoch*num_steps+step)

            model.eval()
            (t, c) = next(iter(test_loader))
            t, c = t.to(device), c.to(device)
            v_t, v_c = model(t, c)
            test_loss = lossfunc(v_t, v_c.to(device))
            (t, c) = next(iter(dev_loader))
            t, c = t.to(device), c.to(device)
            v_t, v_c = model(t, c)
            dev_loss = lossfunc(v_t, v_c.to(device))
            writer.add_scalars('loss', {'train': loss.tolist(),
                                        'test': test_loss.tolist(),
                                        'dev': dev_loss.tolist()
                                       }, epoch*num_steps+step)
            model.train()
    torch.save(model.state_dict(), 'MSE_SGD/epoch_{}.pt'.format(epoch))
