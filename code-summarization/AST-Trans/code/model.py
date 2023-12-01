import torch.nn as nn
import torch
from components import _get_clones, FastRelEmbeddings, FeedForward, SublayerConnection
from fast_attn import FastMultiHeadedAttention
import math
from torch.nn.modules.transformer import _get_activation_fn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class FastASTTrans(nn.Module):
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
    def __init__(self, args, beam_size, max_target_length, tokenizer, vocab_size, hidden_size, par_heads, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout):
        super(FastASTTrans, self).__init__()

        self.beam_size = beam_size
        self.max_length = max_target_length
        self.pad_id = tokenizer.pad_token_id
        self.sos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.num_heads = num_heads
        bro_heads = num_heads - par_heads
        self.pos_type = pos_type.split('_')

        self.word_embedding = Embeddings(hidden_size=args.emb_size,
                                        vocab_size=vocab_size,
                                        dropout=dropout,
                                        with_pos=False)



        encoder_layer = FastASTEncoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout)
        self.encoder = FastASTEncoder(args.max_src_len, encoder_layer, num_layers, par_heads, bro_heads, self.pos_type,
                                      max_rel_pos, hidden_size, dropout=dropout)

        decoder_layer = DecoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout, activation="gelu")
        self.decoder = BaseDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        source_ids = x.source_ids
        memory_key_padding_mask = source_ids.eq(self.pad_id)
        par_edges = x.par_edges
        bro_edges = x.bro_edges
        target_ids = x.target_ids
        target_mask = x.target_mask

        source_embed = self.word_embedding(source_ids)
        encoder_output = self.encoder(source_embed, par_edges, bro_edges)
        encoder_output = encoder_output.permute(1, 0, 2)

        if len(target_ids) != 0:
            tgt_mask = make_std_mask(target_ids, self.pad_id)
            tgt_embeddings = self.word_embedding(target_ids).permute([1, 0, 2]).contiguous()
            tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
            out, attn_weights = self.decoder(tgt=tgt_embeddings, memory=encoder_output,
                                                 tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
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
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = encoder_output[:, i:i + 1]
                context = context.repeat(1, self.beam_size, 1)

                cur_memory_key_padding_mask = memory_key_padding_mask[i:i+1, :]
                cur_memory_key_padding_mask = cur_memory_key_padding_mask.repeat(self.beam_size, 1)

                for _ in range(self.max_length):
                    if beam.done():
                        break

                    tgt_mask = make_std_mask(input_ids, self.pad_id)
                    tgt_embeddings = self.word_embedding(input_ids).permute([1, 0, 2]).contiguous()
                    tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
                    out, attn_weights = self.decoder(tgt=tgt_embeddings, memory=context,
                                                     tgt_mask=tgt_mask, memory_key_padding_mask=cur_memory_key_padding_mask)

                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(
        subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_sequent_mask) != 0


class Embeddings(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout=0.1, with_pos=False):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        if with_pos:
            self.pos_emb = PositionalEncoding(hidden_size)
        else:
            self.pos_emb = None
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        words_embeddings = self.word_embeddings(x)
        if self.pos_emb is not None:
            words_embeddings = self.pos_emb(words_embeddings)

        embeddings = self.norm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) *
                             -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


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
        return output

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


class BaseDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(BaseDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 3)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)



    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt, attn_weights = self.sublayer[0](tgt, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask,
                                                                           key_padding_mask=tgt_key_padding_mask))

        tgt, attn_weights = self.sublayer[1](tgt, lambda x: self.multihead_attn(
            x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        ))

        tgt, _ = self.sublayer[2](tgt, self.feed_forward)
        return tgt, attn_weights
        

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
        
