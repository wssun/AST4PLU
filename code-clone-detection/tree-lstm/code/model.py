import sys

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

        
class Model(nn.Module):   
    def __init__(self, args):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(args.word_vocab_size, args.emb_size, padding_idx=1)
        if args.model == 'child-sum':
            self.encoder = ChildsumLayer(args.hidden_size, args.hidden_size, layer=1, args=args)
        else:
            self.encoder = NaryLayer(args.hidden_size, args.hidden_size, nary=2, layer=1, args=args)
        self.classifier = ClassificationHead(args)
        self.args = args

    def forward(self, input1=None, input2=None, labels=None):
        output1 = self.encoder(input1)
        output2 = self.encoder(input2)

        abs_dist = torch.abs(torch.add(output1, -output2))
        logits = self.classifier(abs_dist)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
        
 
class ChildsumLayer(nn.Module):
    def __init__(self, dim_E, dim_rep, layer=1, args=None):
        super(ChildsumLayer, self).__init__()
        self.layer = layer
        self.args = args
        self.E = TreeEmbeddingLayer(args.word_vocab_size, args=args)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), ChildSumLSTMLayer(dim_E, dim_rep, args))
        print("I am Child-sum model, dim is {} and {} layered".format(str(dim_rep), str(self.layer)))

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
        self.U_f = nn.Linear(dim_out, dim_out)
        self.U_iuo = nn.Linear(dim_out, dim_out * 3)
        self.W = nn.Linear(dim_in, dim_out * 4)
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
        mask = torch.tensor(mask_bool, dtype=torch.float32).to(self.args.device)  # [node_number, child_number]

        index = torch.where(mask_bool, indice, torch.zeros_like(indice))
        h = h_tensor[index]  # [node_number, child_number, dim_out]
        c = c_tensor[index]  # [node_number, child_number, dim_out]

        mask = mask.unsqueeze(-1)
        h_sum = torch.sum(h * mask, 1)  # [node_number, dim_out]

        W_x = self.W(x)  # [node_number, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [node_number, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = torch.reshape(self.U_f(torch.reshape(h, [-1, h.shape[-1]])), h.shape)  # [node_number, child_number, dim_out]
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)  # [node_number, child_number, dim_out]
        branch_f = torch.sum(branch_f_k * c * mask, 1)  # [node_number, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [node_number, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [node_number, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)  # [node_number, dim_out]
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)  # [node_number, dim_out]

        new_c = branch_i * branch_u + branch_f   # [node_number, dim_out]
        new_h = branch_o * torch.tanh(new_c)   # [node_number, dim_out]

        return new_h, new_c


class TreeEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, args):
        super(TreeEmbeddingLayer, self).__init__()
        self.args = args
        self.E = nn.Parameter(torch.randn(vocab_size, args.hidden_size))
        self.linear = nn.Linear(args.hidden_size * args.label_size, args.hidden_size)
        self.shape_size = args.hidden_size * args.label_size


    def forward(self, x):
        x_len = [xx.shape[0] for xx in x]    # x_len = [nodes_number(每层可能不一样)]
        index = torch.cat(x, dim=0).long().view(-1)   # tokenize_id 的一维向量 [sum(nodes_number) * label_size, 1]
        ex = torch.index_select(self.E, dim=0, index=index).view(-1, self.shape_size)  # [sum(nodes_number), label_size * emb_size]
        output = self.linear(ex)   # [sum(nodes_number), emb_size]
        output = torch.split(output, x_len, 0)  # [layer, nodes_number(每层可能不一样), emb_size]
        return output


class NaryLayer(nn.Module):
    def __init__(self, dim_E, dim_rep, nary, layer=1, args=None):
        super(NaryLayer, self).__init__()
        self.layer = layer
        self.nary = nary
        self.args = args
        self.E = TreeEmbeddingLayer(args.word_vocab_size, args=args)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), NaryTreeLSTMLayer(dim_E, dim_rep, nary, args))
        print("I am {}-ary model, dim is {} and {} layered".format(str(nary), str(dim_rep), str(self.layer)))

    def forward(self, x):
        tensor, indice, tree_num = x
        # tensor: n级节点     [layer, nodes_number(每层可能不一样), label_size]
        # indice: n级节点的子节点集合    [layer, nodes_number(每层可能不一样), child_number]
        # tree_num: 该节点属于哪棵树

        tensor = self.E(tensor)   # [layer, nodes_number(每层可能不一样), hidden_size]
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]   # [root_nodes_number, hidden_size]
        return hx


class NaryTreeLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out, nary,  args):
        super(NaryTreeLSTMLayer, self).__init__()
        self.args = args
        self.nary = nary
        self.dim_in = dim_in  # emb_size
        self.dim_out = dim_out
        self.U_f = nn.Linear(nary * dim_out, nary * dim_out)
        self.U_iuo = nn.Linear(nary * dim_out, dim_out * 3)
        self.W = nn.Linear(dim_in, dim_out * 4)
        h_init = torch.zeros([1, dim_out], dtype=torch.float32)
        self.h_init = h_init.to(args.device)
        c_init = torch.zeros([1, dim_out], dtype=torch.float32)
        self.c_init = c_init.to(args.device)

    def forward(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):   # x: [nodes_number, hidden_size]   indice: [nodes_number, child_number]
            h_tensor, c_tensor = self.node_forward(x, h_tensor, c_tensor, indice)
            h_tensor = torch.cat([self.h_init, h_tensor], 0)  # [node_number+1, dim_out]
            c_tensor = torch.cat([self.c_init, c_tensor], 0)  # [node_number+1, dim_out]
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c  # res_h: [layer, nodes_number(每层可能不一样), dim_out]  res_c: [layer, nodes_number(每层可能不一样), dim_out]

    def node_forward(self, x, h_tensor, c_tensor, indice):
        if indice.shape[1] < self.nary:  # child_number < nary
            padding = torch.zeros(indice.shape[0], self.nary-indice.shape[1]).to(self.args.device)
            indice = torch.cat([indice, padding], 1)  # [nodes_number, nary]
        elif indice.shape[1] > self.nary:
            print('Error: child number more than {}!'.format(self.nary))
            sys.exit(0)

        mask_bool = torch.ne(indice, -1)
        mask = torch.tensor(mask_bool, dtype=torch.float32).to(self.args.device)  # [nodes_number, nary]

        index = torch.where(mask_bool, indice, torch.zeros_like(indice)).long() # 把indice中的-1变为0  [nodes_number, nary]
        h = h_tensor[index]  # [nodes_number, nary, dim_out]
        c = c_tensor[index]  # [nodes_number, nary, dim_out]

        W_x = self.W(x)  # [nodes_number, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes_number, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        # torch.reshape(h, [h.shape[0], -1]): [nodes_number, nary * dim_out]
        branch_f_k = torch.reshape(self.U_f(torch.reshape(h, [h.shape[0], -1])), h.shape)  # [nodes_number, nary, dim_out]
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)  # [nodes_number, nary, dim_out]
        mask = mask.unsqueeze(-1)  # [nodes_number, nary, 1]
        branch_f = torch.sum(branch_f_k * c * mask, 1)  # [nodes_number, dim_out]

        # torch.reshape(h, [h.shape[0], -1]): [nodes_number, nary * dim_out]
        branch_iuo = self.U_iuo(torch.reshape(h, [h.shape[0], -1]))  # [nodes_number, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:,:self.dim_out * 1] + W_i_x)  # [nodes_number, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x) # [nodes_number, dim_out]
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x) # [nodes_number, dim_out]

        new_c = branch_i * branch_u + branch_f  # [node_number, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node_number, dim_out]

        return new_h, new_c
