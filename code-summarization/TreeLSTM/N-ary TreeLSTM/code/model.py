# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
import random
from torch.autograd import Variable
import copy
import sys

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.NaryTreelstm = NaryLayer(args.hidden_size, args.hidden_size, nary=2, layer=1, args=args)
        self.decoder = SeqDecoder_BiLSTM(emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.word_embeddings = nn.Embedding(args.word_vocab_size, args.emb_size, padding_idx=1)
        self.args = args
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.word_vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        self.teacher_forcing_ratio = 0.5
        self.linear = nn.Linear(512 * 2, 512)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if True:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.word_embeddings)

    def forward(self, tree_input=None, target_ids=None, target_mask=None):
        encoder_hidden = self.NaryTreelstm(tree_input)
        if target_ids is not None:
            # attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            # tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            # out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())

            tgt_embeddings = self.word_embeddings(target_ids)
            decoder_input = tgt_embeddings[:, 0:1, :]  # shape: (batch_size, input_size)
            decoder_hidden = encoder_hidden
            outputs = torch.zeros(tgt_embeddings.shape[0], self.max_length, tgt_embeddings.shape[2], device="cuda")
            # use teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                for t in range(self.max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output
                    decoder_input = tgt_embeddings[:, t:t + 1, :]
            # predict recursively
            else:
                for t in range(self.max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output  # 32 * 512
                    decoder_input = decoder_output.unsqueeze(1)

            hidden_states = torch.tanh(self.dense(outputs)).contiguous()  # dense input ( 2, 30, 512 )
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(encoder_hidden[0].shape[1]):
                decoder_hidden = (encoder_hidden[0][:, i:i + 1, :].repeat(1, self.beam_size, 1),
                                  encoder_hidden[1][:, i:i + 1, :].repeat(1, self.beam_size, 1))
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()

                for t in range(self.max_length):
                    if beam.done():
                        break
                    decoder_input = self.word_embeddings(input_ids)
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                    hidden_states = torch.tanh(self.dense(decoder_output))
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids = beam.getCurrentState()

                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class SeqEncoder_BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers):
        super(SeqEncoder_BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0,
                            batch_first=True, bidirectional=True, num_layers=n_layers)

    def forward(self, inputs):
        hids, (h_n, c_n) = self.lstm(inputs)

        h_n = torch.sum(h_n, axis=0)  # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        c_n = c_n[0]

        # return hids, (h_n, c_n)
        return (h_n, c_n)


class SeqDecoder_BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers):
        super(SeqDecoder_BiLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0.1,
                            batch_first=True, bidirectional=True, num_layers=n_layers)
        # self.linear = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        hids, (h_n, c_n) = self.lstm(inputs, hidden)
        # hids = self.linear(hids)

        output = torch.sum(h_n, axis=0)  # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        # output = h_n[0]

        # return hids, (h_n, c_n)
        return output, (h_n, c_n)




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

        # hx = tensor[-1]   # [root_nodes_number, hidden_size]

        hx = tensor[-1].unsqueeze(0).repeat(2,1,1)
        cx = c[-1].unsqueeze(0).repeat(2,1,1)
        return hx, cx


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




class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1])
        batch = batch.view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

