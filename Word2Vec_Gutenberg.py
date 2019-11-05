#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import torch
import torch.nn as nn
from nltk.corpus import gutenberg
from torch.utils.data import Dataset, DataLoader

# ### Get all sentences in Project Gutenberg

# In[78]:


# sents = []
# for file in gutenberg.fileids():
#     for sent in gutenberg.sents(file):
#         sents.append(sent)
# len(sents)


# ### Train a word2vec model using gensim for comparison

# In[80]:


# model = gensim.models.Word2Vec(sents, size=100, window=5, min_count=1, workers=4)
# model.save('word2vec_gensim')


# ### Create Vocabulary for all the words in Project Gutenberg

# In[81]:


# words = []
# for file in gutenberg.fileids():
#     for word in gutenberg.words(file):
#         words.append(word)
# # words = list(set(words))
# len(words)


# In[83]:


# fd = nltk.FreqDist(words)
# vocab = sorted(fd, key=fd.get, reverse=True)
# vocab[:10]


# In[84]:


# with open('vocab', 'wb') as f:
#     for i in range(len(vocab)):
#         f.write('{} {}\n'.format(vocab[i], i).encode('utf-8'))


# In[86]:


# with open('vocab.json', 'w') as f:
#     json.dump(vocab, f)


# In[87]:


with open('vocab.json', 'r') as f:
    vocab = json.load(f)


# ###### Since we only have 2.6 million words, Skip-Gram should performs better

# ### Set hyperparameters 

# In[3]:


settings = {
    'vocab_size': len(vocab),
    'window_size': 5,
    'num_epochs': 100,
    'embedding_dim': 50,
    'batch_size': 256,
    'num_heads': 12,
    'dim_head': 64,
    'learning_rate': 1e-5
}


# In[16]:


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
                    context = [self.word2id[word] for word in sent[max(0, i - self.window_size):i] + sent[i+1:min(
                        len(sent), i + 1 + self.window_size)]]
                    target = self.word2id[sent[i]]
                    while len(context) < 2*self.window_size:
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


# In[17]:


dataset = myDataset(settings)
dataloader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True)


# In[62]:


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
#         self.embedding = torch.randn([self.vocab_size, self.embed_dim], requires_grad=True)
        self.W_Q = nn.Linear(self.embed_dim, self.num_hidden)
        self.W_K = nn.Linear(self.embed_dim, self.num_hidden)
        self.W_V = nn.Linear(self.embed_dim, self.num_hidden)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def attention(self, target, context):
        Q = self.W_Q(target).view(self.batch_size, self.num_heads, self.dim_head)
        W = torch.zeros([self.batch_size, self.seq_len, self.num_heads, self.num_heads])
        V = torch.zeros([self.batch_size, self.seq_len, self.num_hidden])
        
        # zero-padding
        for i in range(self.batch_size):
            K_t = self.W_K(target[i]).view(self.num_heads, self.dim_head).transpose(0,1)
            for j in range(self.seq_len):
                W[i][j] = torch.matmul(Q[i], K_t) / (self.dim_head ** 0.5)
                V[i][j] = self.W_V(target[j])
        W = nn.Softmax(dim=-1)(W)
        V = V.view(self.batch_size, self.seq_len, self.num_heads, self.dim_head)
        
        tmp = torch.matmul(W, V).view(self.batch_size, self.seq_len, self.num_hidden)
        context_vector = torch.sum(tmp, dim=1)
        target_vector = self.W_V(target).view(self.batch_size, self.num_hidden)
        return target_vector, context_vector.view(self.batch_size, self.num_hidden)
    
    def forward(self, t, c):
        target = self.embedding(t.long())
        context = self.embedding(c.long())
        v_t, v_c = self.attention(target, context)
        return v_t, v_c
#         sim = self.cos_sim(v_t, v_c)
#         return sim


# In[76]:


model = w2v_model(settings)
lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# In[77]:


for step, (t, c) in enumerate(dataloader):
    optimizer.zero_grad()
    v_t, v_c = model(t, c)
    loss = lossfunc(v_t, v_c)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(loss.data)


# In[ ]:




