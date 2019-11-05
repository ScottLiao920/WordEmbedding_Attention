#!/usr/bin/env python
# coding: utf-8
import json
import os

import gensim
import nltk
import torch
from nltk.corpus import gutenberg
from torch.utils.data import Dataset


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
            model = gensim.models.Word2Vec(self.sents, size=w2v_size, window=window, min_count=min_count, workers=num_workers)
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
        print("Total number of words appeared: {}, including {} unique words.\n".format(len(self.words), len(list(set(self.words)))))

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




with open('vocab.json', 'r') as f:
    vocab = json.load(f)


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
