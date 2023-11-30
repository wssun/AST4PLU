import torch
import torch.nn as nn
from torch.autograd import Variable
from base_data_set import PAD, UNK
from base_data_set import make_std_mask
from components import process_data
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, 2)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BaseTrans(nn.Module):
    def __init__(self, hidden_size):
        super(BaseTrans, self).__init__()
        self.classifier = ClassificationHead(hidden_size)

    def base_process(self, data):
        # process_data(data)

        ast1 = data.ast1
        data.mask1 = ast1.eq(PAD)
        data.emb1 = self.src_embedding(ast1)

        ast2 = data.ast2
        data.mask2 = ast2.eq(PAD)
        data.emb2 = self.src_embedding(ast2)

        # if data.tgt_seq is not None:
        #     tgt_seq = data.tgt_seq
        #     data.tgt_mask = make_std_mask(tgt_seq, PAD)
        #     data.tgt_emb = self.tgt_embedding(tgt_seq)

    def process_data(self, data):
        self.base_process(data)

    def forward(self, data):
        self.process_data(data)

        # ast1 = data.ast1
        # par1 = data.par1
        # bro1 = data.bro1
        # ast2 = data.ast2
        # par2 = data.par2
        # bro2 = data.bro2
        # idx1 = data.idx1
        # idx2 = data.idx2
        # label = data.label
        # decoder_outputs, attn_weights = self.decode(data, encoder_outputs)
        # out = self.generator(decoder_outputs)
        # return out

        output1 = self.encode(ast=data.ast1, emb=data.emb1, par=data.par1, bro=data.bro1)
        output2 = self.encode(ast=data.ast2, emb=data.emb2, par=data.par2, bro=data.bro2)
        abs_dist = torch.abs(torch.add(output1, -output2))
        logits = self.classifier(abs_dist)
        prob = F.softmax(logits)
        return prob

    def encode(self, ast, emb, par, bro):
        x = self.encoder(emb, par, bro)
        x = (x * ast.ne(1)[:, :, None]).sum(1) / ast.ne(1).sum(-1)[:, None]
        return x

    # def decode(self, data, encoder_outputs):
    #     tgt_emb = data.tgt_emb
    #     tgt_mask = data.tgt_mask
    #     src_mask = data.src_mask
    #
    #     tgt_emb = tgt_emb.permute(1, 0, 2)
    #     encoder_outputs = encoder_outputs.permute(1, 0, 2)
    #     tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
    #     outputs, attn_weights = self.decoder(tgt=tgt_emb, memory=encoder_outputs,
    #                                          tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
    #     outputs = outputs.permute(1, 0, 2)
    #     return outputs, attn_weights


# class GreedyGenerator(nn.Module):
#     def __init__(self, model, max_tgt_len, multi_gpu=False):
#         super(GreedyGenerator, self).__init__()
#         if multi_gpu:
#             self.model = model.module
#         else:
#             self.model = model
#         self.max_tgt_len = max_tgt_len
#         self.start_pos = BOS
#         self.unk_pos = UNK
#
#     def forward(self, data):
#         data.tgt_seq = None
#         self.model.process_data(data)
#
#         encoder_outputs = self.model.encode(data)
#
#         batch_size = encoder_outputs.size(0)
#         ys = torch.ones(batch_size, 1).fill_(self.start_pos).long().to(encoder_outputs.device)
#         for i in range(self.max_tgt_len - 1):
#             data.tgt_mask = make_std_mask(ys, 0)
#             data.tgt_emb = self.model.tgt_embedding(Variable(ys))
#             decoder_outputs, decoder_attn = self.model.decode(data, encoder_outputs)
#             out = self.model.generator(decoder_outputs)
#             out = out[:, -1, :]
#             _, next_word = torch.max(out, dim=1)
#             ys = torch.cat([ys,
#                             next_word.unsqueeze(1).long().to(encoder_outputs.device)], dim=1)
#
#         return ys[:, 1:]
