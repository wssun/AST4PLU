# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
import math
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
    def __init__(self, args, beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.code_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6, max_len=args.code_size)
        self.sbt_encoder = SeqEncoder_Transformer(d_model=args.emb_size, nhead=8, num_layers=6, max_len=args.sbt_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.word_embeddings = nn.Embedding(args.vocab_size, args.emb_size, padding_idx=1)

        self.args = args
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.hidden_align = nn.Linear(args.sbt_size, args.code_size)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        self.linear = nn.Linear(512 * 2, 512)
        
        self.beam_size=beam_size
        self.max_length=max_length
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
        
    def forward(self, source_ids=None, source_mask=None,
                source_ids2=None, source_mask2=None, target_ids=None, target_mask=None):
        if self.args.mode == 'token':
            source_embed = self.word_embeddings(source_ids)
            encoder_output = self.code_encoder(source_embed, (1-source_mask).bool()) # 64 2 512
        elif self.args.mode == 'sbt':
            source_embed = self.word_embeddings(source_ids)
            encoder_output = self.sbt_encoder(source_embed, (1-source_mask).bool())
        else:
            token_embed = self.word_embeddings(source_ids)
            token_hidden = self.code_encoder(token_embed, (1-source_mask).bool())
            token_hidden = token_hidden.permute([1, 0, 2])

            sbt_embed = self.word_embeddings(source_ids2)
            sbt_hidden = self.sbt_encoder(sbt_embed, (1-source_mask2).bool())
            sbt_hidden = sbt_hidden.permute([2, 1, 0])
            sbt_hidden = self.hidden_align(sbt_hidden)
            sbt_hidden = sbt_hidden.permute([1, 2, 0])

            encoder_output = torch.cat([token_hidden, sbt_hidden], dim=2)
            encoder_output = self.linear(encoder_output)
            encoder_output = encoder_output.permute([1, 0, 2])


        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])  # 32 32
            tgt_embeddings = self.word_embeddings(target_ids).permute([1,0,2]).contiguous()  # 32 2 768
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # 2 32 768
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)     
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.word_embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds


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



        

class Beam(object):
    def __init__(self, size,sos,eos):
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
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
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
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
