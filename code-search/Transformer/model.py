# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss



class Model(nn.Module):
    def __init__(self, tokenizer,args):
        super(Model, self).__init__()
        self.code_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6, max_len=args.code_size)
        self.query_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6, max_len=args.query_size)
        self.sbt_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6, max_len=args.sbt_size)
        self.word_embedding = nn.Embedding(args.vocab_size, args.emb_size, padding_idx=1)
        self.args = args
        
    def forward(self, token_inputs=None, sbt_inputs=None, nl_inputs=None, mode="sbt"):
        batch_size = nl_inputs.shape[0]

        if mode == "token":
            token_embed = self.word_embedding(token_inputs)
            code_vec = self.code_encoder(token_embed, token_inputs.eq(1))
            code_vec = (code_vec * token_inputs.ne(1)[:, :, None]).sum(1) / token_inputs.ne(1).sum(-1)[:, None]
        elif mode == "sbt":
            sbt_embed = self.word_embedding(sbt_inputs)
            code_vec = self.sbt_encoder(sbt_embed, sbt_inputs.eq(1))
            code_vec = (code_vec * sbt_inputs.ne(1)[:, :, None]).sum(1) / sbt_inputs.ne(1).sum(-1)[:, None]
        else:
            token_embed = self.word_embedding(token_inputs)
            sbt_embed = self.word_embedding(sbt_inputs)

            token_vec = self.code_encoder(token_embed, token_inputs.eq(1))
            sbt_vec = self.sbt_encoder(sbt_embed, sbt_inputs.eq(1))

            token_vec = (token_vec * token_inputs.ne(1)[:, :, None]).sum(1) / token_inputs.ne(1).sum(-1)[:, None]
            sbt_vec = (sbt_vec * sbt_inputs.ne(1)[:, :, None]).sum(1) / sbt_inputs.ne(1).sum(-1)[:, None]

            # code_vec = token_vec + sbt_vec
            code_vec = torch.cat([token_vec, sbt_vec], dim=1)
        nl_embed = self.word_embedding(nl_inputs)
        nl_vec = self.query_encoder(nl_embed, nl_inputs.eq(1))
        nl_vec = (nl_vec * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]

        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(batch_size, device=scores.device))
        return loss, code_vec, nl_vec


class SeqEncoder_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len):
        super(SeqEncoder_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_embedding = PositionalEncoding(d_model, 0.1, max_len)

    def forward(self, inputs, mask):
        input_embedding = self.pos_embedding(inputs)
        input_embedding = input_embedding.permute(1, 0, 2)

        outputs = self.encoder(input_embedding, src_key_padding_mask=mask)
        outputs = outputs.permute(1, 0, 2)

        return outputs
        
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]
        return self.dropout(x)