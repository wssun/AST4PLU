# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
import random
from torch.autograd import Variable
import copy


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
        self.token_encoder = SeqEncoder_BiLSTM(emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.sbt_encoder = SeqEncoder_BiLSTM(emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.decoder = SeqDecoder_BiLSTM(emb_size=args.emb_size, hidden_size=512, n_layers=1)
        self.word_embeddings = nn.Embedding(args.vocab_size, args.emb_size, padding_idx=1)
        self.args = args
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
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

    def forward(self, source_ids=None, source_ids2=None, target_ids=None, target_mask=None):
        if self.args.mode == 'token':
            source_embed = self.word_embeddings(source_ids)
            encoder_hidden = self.token_encoder(source_embed)
        elif self.args.mode == 'sbt':
            source_embed = self.word_embeddings(source_ids)
            encoder_hidden = self.sbt_encoder(source_embed)
        else:
            token_embed = self.word_embeddings(source_ids)
            token_hidden = self.token_encoder(token_embed)

            sbt_embed = self.word_embeddings(source_ids2)
            sbt_hidden = self.token_encoder(sbt_embed)

            h0 = torch.cat([token_hidden[0], sbt_hidden[0]], dim=2)
            h0 = self.linear(h0)

            h1 = torch.cat([token_hidden[1], sbt_hidden[1]], dim=2)
            h1 = self.linear(h1)
            encoder_hidden = (h0,h1)

        if target_ids is not None:

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
                    outputs[:, t, :] = decoder_output
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
            for i in range(source_ids.shape[0]):
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

        # h_n = torch.sum(h_n, axis=0)  # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        # c_n = c_n[0]

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
        # c_n = c_n[0]

        # return hids, (h_n, c_n)
        return output, (h_n, c_n)


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

