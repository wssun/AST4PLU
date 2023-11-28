# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, 2)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, args, tokenizer):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.emb_size, padding_idx=tokenizer.pad_token_id)
        self.encoder = SeqEncoder_BiLSTM(emb_size=args.emb_size, hidden_size=args.emb_size, n_layers=1)
        self.fully_connect = nn.Linear(3*args.emb_size, args.hidden_size)
        # self.attention = nn.Linear(args.hidden_size, 1)
        self.classifier = ClassificationHead(args)
        self.args = args

    def forward(self, left1=None, p1=None, right1=None, left2=None, p2=None, right2=None, labels=None):
        left1 = self.embedding(left1)     # batch * sample_contexts * token_size * emb_size
        p1 = self.embedding(p1)           # batch * sample_contexts * path_size * emb_size
        p1 = p1.view(-1, self.args.path_size, self.args.emb_size)  # (batch * sample_contexts) * path_size * emb_size
        right1 = self.embedding(right1)   # batch * sample_contexts * token_size * emb_size

        left1 = torch.sum(left1,dim=2)        # batch * sample_contexts * emb_size
        _, (p1, _) = self.encoder(p1)     # (batch * sample_contexts) * emb_size
        p1 = p1.view(-1, self.args.sample_contexts, self.args.emb_size)  # batch * sample_contexts * emb_size
        right1 = torch.sum(right1,dim=2)      # batch * sample_contexts * emb_size
        path1 = torch.cat([left1, p1, right1], dim=2)  # batch * sample_contexts * (3*emb_size)
        path1 = self.fully_connect(path1)   # batch * sample_contexts * hidden_size
        path1 = torch.tanh(path1)         # batch * sample_contexts * hidden_size
        path1 = torch.mean(path1, dim=1)  # batch * hidden_size

        # attention
        # a = self.attention(path1)    # batch * sample_contexts * 1
        # w = F.softmax(a,dim=2)   # batch * sample_contexts * 1
        # path1 = torch.mul(w, path1)  # batch * sample_contexts * hidden_size
        # path1 = torch.sum(path1, dim=1)  # batch * hidden_size

        left2 = self.embedding(left2)     # batch * sample_contexts * token_size * emb_size
        p2 = self.embedding(p2)           # batch * sample_contexts * path_size * emb_size
        p2 = p2.view(-1, self.args.path_size, self.args.emb_size)  # (batch * sample_contexts) * path_size * emb_size
        right2 = self.embedding(right2)   # batch * sample_contexts * token_size * emb_size

        left2 = torch.sum(left2,dim=2)        # batch * sample_contexts * emb_size
        _, (p2, _) = self.encoder(p2)     # (batch * sample_contexts) * emb_size
        p2 = p2.view(-1, self.args.sample_contexts, self.args.emb_size)  # batch * sample_contexts * emb_size
        right2 = torch.sum(right2,dim=2)      # batch * sample_contexts * emb_size
        path2 = torch.cat([left2, p2, right2], dim=2)  # batch * sample_contexts * (3*emb_size)
        path2 = self.fully_connect(path2)   # batch * sample_contexts * hidden_size
        path2 = torch.tanh(path2)         # batch * sample_contexts * hidden_size
        path2 = torch.mean(path2, dim=1)  # batch * hidden_size

        abs_dist = torch.abs(torch.add(path1, path2))
        logits = self.classifier(abs_dist)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class SeqEncoder_BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers):
        super(SeqEncoder_BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers * 2
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=True, num_layers=n_layers)

    def forward(self, inputs):
        hids, (h_n, c_n) = self.lstm(inputs)

        h_n = torch.sum(h_n, axis=0)
        c_n = c_n[0]

        return hids, (h_n, c_n)




