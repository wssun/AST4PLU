import json
import random

import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm
from torch.utils.data.dataset import T_co
from transformers import RobertaTokenizer
punc = string.punctuation

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class BaseASTDataSet(data.Dataset):
    def __init__(self, args, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        # data_dir = config.data_dir + '/' + data_set_name + '/'
        data_dir = args.data_dir

        self.ignore_more_than_k = args.is_ignore
        self.max_rel_pos = args.max_rel_pos
        self.max_src_len = args.max_src_len
        # self.max_tgt_len = config.max_tgt_len

        ast_path = data_dir + 'un_split_{}.jsonl'.format(args.data_type)
        matrices_path = data_dir + 'un_split_matrices.npz'

        self.ast_data, labels = load_ast(ast_path)
        self.matrices_data = load_matrices(matrices_path)
        all_label_data = load_label(data_dir + data_set_name + '.txt')


        final_data = []
        for item in all_label_data:
            idx1, idx2, label = item
            if idx1 not in labels or idx2 not in labels:
                continue
            else:
                final_data.append([idx1, idx2, label])
        self.label_data = final_data

        self.data_len = len(self.ast_data)   # 9142
        self.data_set_len = len(self.label_data)
        print('data set len:{}'.format(self.data_set_len))
        self.vocab_size = args.vocab_size
        self.collector = Collater([], [])

        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        return word2tensor(ast_seq, self.max_src_len, self.tokenizer)

    # def convert_nl_to_tensor(self, nl):
    #     nl = nl[:self.max_tgt_len - 2]
    #     nl = ['<s>'] + nl + ['</s>']
    #     return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


def word2tensor(seq, max_seq_len, tokenizer):
    # seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    # seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    # seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    seq_tok = tokenizer.tokenize(seq)
    seq_tok = seq_tok[:max_seq_len]

    seq_ids = tokenizer.convert_tokens_to_ids(seq_tok)
    padding_length = max_seq_len - len(seq_ids)
    seq_ids += [tokenizer.pad_token_id] * padding_length
    return torch.tensor(seq_ids, dtype=torch.long)


def load_ast(file_path):
    _data = []
    labels = []
    print(f'loading {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            js = json.loads(line)
            _data.append(js)
            labels.append(js['idx'])
    return _data, labels


def load_seq(file_path):
    data_ = []
    print(f'loading {file_path} ...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def load_matrices(file_path):
    print('loading matrices...')
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


def load_label(file_path):
    _data = []
    print(f'loading {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            x, y, l = line.split()
            _data.append([x,y,l])
    return _data


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



