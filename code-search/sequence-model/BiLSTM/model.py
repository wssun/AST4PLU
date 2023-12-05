# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self, tokenizer, args):
        super(Model, self).__init__()
        self.code_encoder = SeqEncoder_BiLSTM(
            emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.query_encoder = SeqEncoder_BiLSTM(
            emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.sbt_encoder = SeqEncoder_BiLSTM(
            emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.embedding = nn.Embedding(
            args.vocab_size, args.emb_size, padding_idx=1)
        self.args = args
        self.code_bn = nn.BatchNorm1d(
            num_features=512, eps=1e-5, momentum=1, affine=True, track_running_stats=False)
        self.doc_bn = nn.BatchNorm1d(
            num_features=512, eps=1e-5, momentum=1, affine=True, track_running_stats=False)

    def forward(self, token_inputs=None, sbt_inputs=None, nl_inputs=None, mode="sbt"):
        batch_size = nl_inputs.shape[0]

        if mode == "token":
            token_inputs = self.embedding(token_inputs)
            hidden = self.code_encoder.init_hidden(batch_size)
            _, (code_vec, _) = self.code_encoder(token_inputs, hidden)
            code_vec = self.code_bn(code_vec)
        elif mode == "sbt":
            sbt_inputs = self.embedding(sbt_inputs)
            hidden = self.sbt_encoder.init_hidden(batch_size)
            _, (code_vec, _) = self.sbt_encoder(sbt_inputs, hidden)
            code_vec = self.code_bn(code_vec)
        else:
            token_inputs = self.embedding(token_inputs)
            hidden = self.code_encoder.init_hidden(batch_size)
            _, (token_vec, _) = self.code_encoder(token_inputs, hidden)

            sbt_inputs = self.embedding(sbt_inputs)
            hidden = self.sbt_encoder.init_hidden(batch_size)
            _, (sbt_vec, _) = self.sbt_encoder(sbt_inputs, hidden)
            code_vec = torch.cat([token_vec, sbt_vec], dim=1)
            code_vec = self.code_bn(code_vec)

        nl_inputs = self.embedding(nl_inputs)
        nl_hidden = self.query_encoder.init_hidden(batch_size)
        _, (nl_vec, _) = self.query_encoder(nl_inputs, nl_hidden)
        nl_vec = self.doc_bn(nl_vec)

        scores = (nl_vec[:, None, :]*code_vec[None, :, :]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(batch_size, device=scores.device))
        return loss, code_vec, nl_vec


class SeqEncoder_BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers):
        super(SeqEncoder_BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers*2
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0,
                            batch_first=True, bidirectional=True, num_layers=n_layers)

    def init_hidden(self, batch_size):
        # weight = next(self.parameters()).data
        weight = 0
        weight = torch.tensor(weight, dtype=torch.float32)
        weight = weight.cuda()
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_(),  # rnn_type == 'LSTM'
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_())

    def forward(self, inputs, hidden=None):
        hids, (h_n, c_n) = self.lstm(inputs, hidden)

        h_n = torch.sum(h_n, axis=0)  # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        c_n = c_n[0]

        return hids, (h_n, c_n)
