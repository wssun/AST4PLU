# Copyright (c) Microsoft Corporation.

import torch.nn as nn
import torch
import math
from torch.nn import CrossEntropyLoss, MSELoss
from components import _get_clones, FastRelEmbeddings, FeedForward, SublayerConnection
from fast_attn import FastMultiHeadedAttention


class Model(nn.Module):
    def __init__(self, args, hidden_size, par_heads, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout):
        super(Model, self).__init__()
        self.pos_type = pos_type.split('_')
        self.num_heads = num_heads
        bro_heads = num_heads - par_heads
        encoder_layer = FastASTEncoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout)
        self.ast_encoder = FastASTEncoder(args.max_src_len, encoder_layer, num_layers, par_heads, bro_heads, self.pos_type,
                                      max_rel_pos, hidden_size, dropout=dropout)
        self.query_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=self.num_heads, num_layers=6, max_len=args.query_size)
        self.embedding = nn.Embedding(args.vocab_size, args.emb_size, padding_idx=1)

        self.args = args

    def forward(self, x):
        ast_ids = x.ast_ids
        par_edges = x.par_edges
        bro_edges = x.bro_edges
        nl_ids = x.nl_ids
        nl_mask = x.nl_mask
        batch_size = nl_ids.shape[0]

        ast_embed = self.embedding(ast_ids)
        ast_vec = self.ast_encoder(ast_embed, par_edges, bro_edges)

        nl_embed = self.embedding(nl_ids)
        nl_vec = self.query_encoder(nl_embed, nl_ids.eq(1))
        nl_vec = (nl_vec * nl_ids.ne(1)[:, :, None]).sum(1) / nl_ids.ne(1).sum(-1)[:, None]

        scores = (nl_vec[:, None, :]*ast_vec[None, :, :]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(batch_size, device=scores.device))
        return loss, ast_vec, nl_vec




class FastASTEncoder(nn.Module):
    def __init__(self, max_src_len, encoder_layer, num_layers, par_heads, bro_heads, pos_type, max_rel_pos,
                 hidden_size, dropout=0.2):
        super(FastASTEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=(max_src_len, 1), stride=1)
        self.par_heads = par_heads
        self.bro_heads = bro_heads
        d_k = hidden_size // (par_heads + bro_heads)
        if par_heads > 0:
            self.par_rel_emb = FastRelEmbeddings(d_k, par_heads, max_rel_pos, pos_type, dropout=dropout)
        if bro_heads > 0:
            self.bro_rel_emb = FastRelEmbeddings(d_k, bro_heads, max_rel_pos, pos_type, dropout=dropout)

        self.end_nodes = None

    def forward(self, src_emb, par_edges, bro_edges):
        output = src_emb
        rel_par_pos = par_edges
        rel_bro_pos = bro_edges

        batch_size, max_rel_pos, max_ast_len = rel_par_pos.size()
        rel_par_q, rel_par_k, rel_par_v = None, None, None
        rel_bro_q, rel_bro_k, rel_bro_v = None, None, None
        if self.par_heads > 0:
            rel_par_q, rel_par_k, rel_par_v = self.par_rel_emb()
        if self.bro_heads > 0:
            rel_bro_q, rel_bro_k, rel_bro_v = self.bro_rel_emb()
        rel_q = self.concat_vec(rel_par_q, rel_bro_q, dim=1)
        rel_k = self.concat_vec(rel_par_k, rel_bro_k, dim=1)
        rel_v = self.concat_vec(rel_par_v, rel_bro_v, dim=1)

        start_nodes = self.concat_pos(rel_par_pos, rel_bro_pos)

        need_end_nodes = True
        if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
            need_end_nodes = False

        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.end_nodes = end_nodes.repeat(batch_size, self.par_heads + self.bro_heads,
                                              max_rel_pos, 1)

        for i, layer in enumerate(self.layers):
            output = layer(output, start_nodes, self.end_nodes, rel_q, rel_k, rel_v)

        output = self.norm(output)
        output = self.MaxPool2d(output)
        return output.squeeze(1)

    def concat_pos(self, rel_par_pos, rel_bro_pos):
        if self.par_heads == 0:
            return rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                              dim=1)
        if self.bro_heads == 0:
            return rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                              dim=1)

        rel_par_pos = rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                                 dim=1)
        rel_bro_pos = rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                                 dim=1)
        rel_pos = self.concat_vec(rel_par_pos, rel_bro_pos, dim=1)

        return rel_pos

    @staticmethod
    def concat_vec(vec1, vec2, dim):
        if vec1 is None:
            return vec2
        if vec2 is None:
            return vec1
        return torch.cat([vec1, vec2], dim=dim)


class FastASTEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feed_forward, dropout):
        super(FastASTEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attn = FastMultiHeadedAttention(num_heads, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dim_feed_forward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.sublayer = _get_clones(SublayerConnection(hidden_size, dropout), 2)

    def forward(self, src, start_nodes, end_nodes, rel_q, rel_k, rel_v):
        src, attn_weights = self.sublayer[0](src, lambda x: self.self_attn(x, x, x, start_nodes, end_nodes,
                                                                           rel_q, rel_k, rel_v))
        src, _ = self.sublayer[1](src, self.feed_forward)
        return src


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
    def __init__(self, d_model, dropout=0.1, max_len=2501):
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
        x = x + self.pe

        return self.dropout(x)