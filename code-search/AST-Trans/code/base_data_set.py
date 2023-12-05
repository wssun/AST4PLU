import json
import random

import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data.dataloader import Collater
from torch.utils.data.dataset import T_co
from transformers import RobertaTokenizer
punc = string.punctuation


class BaseASTDataSet(data.Dataset):
    def __init__(self, args, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = args.data_dir + '/' + data_set_name + '/'

        self.ignore_more_than_k = args.is_ignore
        self.max_rel_pos = args.max_rel_pos
        self.max_src_len = args.max_src_len
        self.max_nl_len = args.query_size

        ast_path = data_dir + 'un_split_{}.jsonl'.format(args.data_type)
        matrices_path = data_dir + 'un_split_matrices.npz'

        self.ast_data = load_ast(ast_path)
        self.matrices_data = load_matrices(matrices_path)

        self.data_set_len = len(self.ast_data)
        print('data set len:{}'.format(self.data_set_len))
        self.vocab_size = args.vocab_size
        self.collector = Collater([], [])

        self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq, tokenizer):
        seq_tok = tokenizer.tokenize(ast_seq)
        seq_tok = seq_tok[:self.max_src_len]

        seq_ids = tokenizer.convert_tokens_to_ids(seq_tok)
        padding_length = self.max_src_len - len(seq_ids)
        seq_ids += [tokenizer.pad_token_id] * padding_length
        return torch.tensor(seq_ids, dtype=torch.long)

    def convert_nl_to_tensor(self, nl, tokenizer):
        target_tokens = tokenizer.tokenize(nl)[:self.max_nl_len - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]

        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = self.max_nl_len - len(target_ids)

        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        return torch.tensor(target_ids, dtype=torch.long), torch.tensor(target_mask, dtype=torch.long)



class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.ast = source
        self.nl = target


def load_ast(file_path):
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            ast = js['ast']
            nl = js['nl']

            examples.append(
                Example(
                    idx=idx,
                    source=ast,
                    target=nl,
                )
            )
    return examples



def load_matrices(file_path):
    print('loading matrices...')
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


def clean_nl(s):
    s = s.strip()
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    processed_words = []
    for w in s.split():
        if w not in punc:
            processed_words.extend(wordninja.split(w))
        else:
            processed_words.append(w)
    return processed_words



def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(
        subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask



