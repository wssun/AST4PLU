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
        self.encoder = SeqEncoder_BiLSTM(emb_size=args.emb_size, hidden_size=args.hidden_size, n_layers=1)
        self.classifier = ClassificationHead(args)
        self.input_type = args.input_type
        self.args = args
        self.linear = nn.Linear(512*2, 512)

    def forward(self, input1_ids=None, input2_ids=None, labels=None):
        if self.input_type == 'mix':
            token1 = self.embedding(input1_ids[0])
            sbt1 = self.embedding(input1_ids[1])
            _, (token_output_1, _) = self.encoder(token1)
            _, (sbt_output_1, _) = self.encoder(sbt1)

            output1 = torch.cat([token_output_1, sbt_output_1], dim=1)
            output1 = self.linear(output1)
            # output1 = token_output_1 + sbt_output_1

            token2 = self.embedding(input2_ids[0])
            sbt2 = self.embedding(input2_ids[1])
            _, (token_output_2, _) = self.encoder(token2)
            _, (sbt_output_2, _) = self.encoder(sbt2)

            output2 = torch.cat([token_output_2, sbt_output_2], dim=1)
            output2 = self.linear(output2)
            # output2 = token_output_2 + sbt_output_2
        else:
            input1_ids = self.embedding(input1_ids)
            input2_ids = self.embedding(input2_ids)

            _, (output1, _) = self.encoder(input1_ids)
            _, (output2, _) = self.encoder(input2_ids)

        abs_dist = torch.abs(torch.add(output1, -output2))
        logits = self.classifier(abs_dist)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob


class SeqEncoder_BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers):
        super(SeqEncoder_BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers*2
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=True, num_layers=n_layers)

    def forward(self, inputs):
        hids, (h_n, c_n) = self.lstm(inputs)

        h_n = torch.sum(h_n, axis=0)
        c_n = c_n[0]

        return hids, (h_n, c_n)


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6,
                                              vocab_size=args.vocab_size)
        self.classifier = ClassificationHead(args)
        self.input_type = args.input_type
        self.linear = nn.Linear(512 * 2, 512)
        self.args = args

    def forward(self, input1_ids=None, input2_ids=None, labels=None):
        if self.input_type == 'mix':
            token1 = self.encoder(input1_ids[0])
            sbt1 = self.encoder(input1_ids[1])
            output1 = torch.cat([token1, sbt1], dim=1)
            output1 = self.linear(output1)
            # output1 = token1 + sbt1

            token2 = self.encoder(input2_ids[0])
            sbt2 = self.encoder(input2_ids[1])
            output2 = torch.cat([token2, sbt2], dim=1)
            output2 = self.linear(output2)
            # output2 = token2 + sbt2
        else:
            output1 = self.encoder(input1_ids)
            output2 = self.encoder(input2_ids)

        abs_dist = torch.abs(torch.add(output1, -output2))
        logits = self.classifier(abs_dist)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class SeqEncoder_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size):
        super(SeqEncoder_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_embedding = PositionalEncoding(d_model, 0.1)
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)

    def forward(self, inputs):
        input_embedding = self.word_embedding(inputs)
        input_embedding = self.pos_embedding(input_embedding)
        input_embedding = input_embedding.permute(1, 0, 2)

        outputs = self.encoder(input_embedding, src_key_padding_mask=inputs.eq(1))
        outputs = outputs.permute(1, 0, 2)
        outputs = (outputs * inputs.ne(1)[:, :, None]).sum(1) / inputs.ne(1).sum(-1)[:, None]

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):  # ninp, dropout
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000 * 200
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)  # batch * sentence len * 512
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size()[0], :]
        x = x + self.pe[:, :x.size()[1], :]  # pe在第二维截取前 sentence len 个
        return self.dropout(x)
 
        


