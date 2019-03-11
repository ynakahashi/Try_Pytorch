#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:56:26 2019

@author: yn250006
"""

""" Set current directory """
cd /Users/yn250006/Documents/GitHub/Try_Pytorch/


""" Load libraries """
import janome
from janome.tokenizer import Tokenizer

import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import FastText

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

torch.manual_seed(0)
torch.cuda.manual_seed(0)

""" Define Tokenizer """
j_t = Tokenizer()
def tokenizer(text): 
    return [tok for tok in j_t.tokenize(text, wakati=True)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False)

""" Read data under the above definition """
# train, val, test = data.TabularDataset.splits(
#         path='./Data/', train='train.tsv',
#         validation='valid.tsv', test='test.tsv', format='tsv',
#         fields=[('Text', TEXT), ('Label', LABEL)])
train, val, test = data.TabularDataset.splits(
        path='./Data/', train='train_small.tsv',
        validation='valid_small.tsv', test='test_small.tsv', format='tsv',
        fields = [('Text', TEXT), ('Label', LABEL)])

""" Check """"
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

print('len(val)', len(val))
print('vars(val[0])', vars(val[0]))

print('len(test)', len(test))
print('vars(test[0])', vars(test[0]))


""" 読み込んだデータに出現した単語のリストを作成し、単語に番号を振る """
TEXT.build_vocab(train, min_freq=10)
""" FastTextの学習済みベクトルを使うならこちら """
# TEXT.build_vocab(train, vectors = FastText(language="ja"), min_freq = 10)


""" Check """
""" dictionaryはsliceできないけど、一度listに変換すればsliceできる """
# TEXT.vocab.freqs
dict(list(TEXT.vocab.freqs.items())[0:10])
# TEXT.vocab.stoi
dict(list(TEXT.vocab.stoi.items())[0:10])
# TEXT.vocab.itos
dict(list(TEXT.vocab.stoi.items())[0:10])


"""
data.Iterator.splits は dataset オブジェクトから、各単語を番号に変換して
ミニバッチごとにまとめた行列を返すイテレータを作成できます。
"""

train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), batch_sizes=(32, 256, 256), device=-1, repeat=False,sort=False)


train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), batch_sizes=(32, 2, 1), device=-1, repeat=False,sort=False)


"""
イテレータの返す結果
batch.Textは[32, 1475]の二次元配列と[32]の一次元配列を含むtuple、
batch.Labelは[32]の一次元配列
"""
batch = next(iter(train_iter))
print(batch.Text)
print(batch.Label)
batch.Text[0].size()
batch.Text[1].size()

# class EncoderRNN(nn.Module):
#     def __init__(self, emb_dim, h_dim, v_size, gpu=True, batch_first=True):
#         super(EncoderRNN, self).__init__()
#         self.gpu = gpu
#         self.h_dim = h_dim
#         self.embed = nn.Embedding(v_size, emb_dim)
#         self.embed.weight.data.copy_(TEXT.vocab.vectors)
#         self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
#                             bidirectional=True)

class EncoderRNN(nn.Module):
    def __init__(self, emb_dim, h_dim, v_size, gpu=True, v_vec=None, batch_first=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        if v_vec is not None:
            self.embed.weight.data.copy_(v_vec)
        self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=True)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.embed(sentence)
        packed_emb = emb

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        out, hidden = self.lstm(packed_emb, self.hidden)

        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(output)[0]

        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

        return out

class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn = f
        self.main = nn.Linear(h_dim, c_num)


    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs) #(b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1) # (b, s, h) -> (b, h)
        return F.log_softmax(self.main(feats),dim=1), attns

def train_model(epoch, train_iter, optimizer, log_interval=1, batch_size=2):
    encoder.train()
    classifier.train()
    correct = 0
    for idx, batch in enumerate(train_iter):
        (x, x_l), y = batch.Text, batch.Label
        optimizer.zero_grad()
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        if idx % log_interval == 0:
            print('train epoch: {} [{}/{}], acc:{}, loss:{}'.format(
            epoch, (idx+1)*len(x), len(train_iter)*batch_size,
            correct/float(log_interval * len(x)),
            loss.data[0]))
            correct = 0

            
def test_model(epoch, test_iter):
    encoder.eval()
    classifier.eval()
    correct = 0
    for idx, batch in enumerate(test_iter):
        (x, x_l), y = batch.Text, batch.Label
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
    print('test epoch:{}, acc:{}'.format(epoch, correct/float(len(test))))



emb_dim = 300 #単語埋め込み次元
h_dim = 3 #lstmの隠れ層の次元
class_num = 2 #予測クラス数
lr = 0.001 #学習係数
epochs = 50 #エポック数

 # make model
encoder = EncoderRNN(emb_dim, h_dim, len(TEXT.vocab),gpu=False, v_vec = TEXT.vocab.vectors)
classifier = AttnClassifier(h_dim, class_num)

# init model
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

for m in encoder.modules():
    print(m.__class__.__name__)
    weights_init(m)
    
for m in classifier.modules():
    print(m.__class__.__name__)
    weights_init(m)

# optim
from itertools import chain
optimizer = optim.Adam(chain(encoder.parameters(),classifier.parameters()), lr=lr)


# train model
for epoch in range(epochs):
    train_model(epoch + 1, train_iter, optimizer)
    test_model(epoch + 1, val_iter)




