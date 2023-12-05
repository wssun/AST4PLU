# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss



class Model(nn.Module):   
    def __init__(self, tokenizer,args):
        super(Model, self).__init__()
        self.code_encoder = SeqEncoder_BiLSTM(args=args, emb_size=512, hidden_size=512, n_layers=1)
        self.query_encoder = SeqEncoder_BiLSTM(args=args, emb_size=512, hidden_size=512, n_layers=1)
        self.childsumtreelstm = ChildsumLayer(512, 512, 512, layer=1, args=args)
        self.embedding = nn.Embedding(150000, 512, padding_idx=1)
        self.args = args
        
    def forward(self, token_inputs=None, tree=None, nl_inputs=None):
        batch_size = nl_inputs.shape[0]
        code_vec = self.childsumtreelstm(tree)

        nl_inputs = self.embedding(nl_inputs)
        _, (nl_vec, _) = self.query_encoder(nl_inputs)

        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(batch_size, device=scores.device))
        return loss,code_vec,nl_vec


class SeqEncoder_BiLSTM(nn.Module):
    def __init__(self, args, emb_size, hidden_size, n_layers):
        super(SeqEncoder_BiLSTM, self).__init__()
        self.args = args
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=True, num_layers=n_layers)

    def forward(self, inputs, hidden=None):
        # input: [batch_sz * 1 * seq_len]
        # hids:[b x 1 x hid_sz*2](biRNN)
        hids, (h_n, c_n) = self.lstm(inputs)

        h_n = torch.sum(h_n, axis=0)  # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        c_n = c_n[0]

        return hids, (h_n, c_n)
        

class ChildsumLayer(nn.Module):
    def __init__(self, dim_E, dim_rep, in_vocab, layer=1, args=None):
        super(ChildsumLayer, self).__init__()
        self.layer = layer
        self.args = args
        self.E = TreeEmbeddingLayer(args.vocab_size, args=args)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), ChildSumLSTMLayer(dim_E, dim_rep, args))
        print("I am Child-sum model, dim is {} and {} layered".format(
            str(dim_rep), str(self.layer)))

    def forward(self, x):
        tensor, indice, tree_num = x
        # tensor: n级节点
        # indice: n级节点的子节点集合
        # tree_num: 该节点属于哪棵树

        tensor = self.E(tensor)
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        return hx


class ChildSumLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out, args):
        super(ChildSumLSTMLayer, self).__init__()
        self.args = args
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = nn.Linear(512, dim_out)
        self.U_iuo = nn.Linear(512, dim_out * 3)
        self.W = nn.Linear(512, dim_out * 4)
        h_init=torch.zeros([1, dim_out], dtype=torch.float32)
        self.h_init = h_init.to(args.device)
        c_init=torch.zeros([1, dim_out], dtype=torch.float32)
        self.c_init = c_init.to(args.device)

    def forward(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.node_forward(x, h_tensor, c_tensor, indice)
            h_tensor = torch.cat([self.h_init, h_tensor], 0)
            c_tensor = torch.cat([self.c_init, c_tensor], 0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def node_forward(self, x, h_tensor, c_tensor, indice):
        mask_bool = torch.ne(indice, -1)
        mask = torch.tensor(mask_bool, dtype=torch.float32)  # [batch, child]

        index = torch.where(mask_bool, indice, torch.zeros_like(indice))
        h = h_tensor[index]  # [nodes, child, dim]
        c = c_tensor[index]

        mask = mask.unsqueeze(-1)
        h_sum = torch.sum(h * mask, 1)  # [nodes, dim_out]
        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = torch.reshape(self.U_f(torch.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = torch.sum(branch_f_k * c * mask, 1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        return new_h, new_c


class TreeEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, args):
        super(TreeEmbeddingLayer, self).__init__()
        self.args = args
        self.E = nn.Parameter(torch.randn(vocab_size, 512))
        self.linear = nn.Linear(512 * args.label_size, 512)
        self.shape_size = 512 * args.label_size


    def forward(self, x):
        x_len = [xx.shape[0] for xx in x]
        index = torch.cat(x, dim=0).long().view(-1)
        ex = torch.index_select(self.E, dim=0, index=index).view(-1, self.shape_size)
        output = self.linear(ex)
        output = torch.split(output, x_len, 0)
        return output