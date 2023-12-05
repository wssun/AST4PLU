# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
from collections import defaultdict, Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import json

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import sys
from tqdm import tqdm
import multiprocessing
from model import Model

cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaTokenizer)
import torch.optim as optim
from time import time
import setproctitle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# pad_id = tokenizer.pad_token_id  # 1
# unk_id = tokenizer.unk_token_id  # 3

logger = logging.getLogger(__name__)
fh = logging.FileHandler(f'logs.txt')
logger.addHandler(fh)  # add the handlers to the logger

sys.setrecursionlimit(2000)
UNK_ID = 3
PAD_ID = 1


def time_format(sec):
    hour = sec // 3600
    sec = sec % 3600
    minute = sec // 60
    second = sec % 60
    return hour, minute, second


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_ids,
                 tree_root,
                 nl_ids,
                 ):
        self.code_ids = code_ids
        self.tree_root = tree_root
        self.nl_ids = nl_ids


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return (res)


def consult_tree(root, tokenizer):
    nodes = traverse(root)
    for n in nodes:
        n.label = tokenizer.convert_tokens_to_ids(n.label)
    return nodes[0]


def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        seq = seq + [-1] * (maxlen - len(seq))

    return seq


def get_nums(roots, device=torch.device('cuda')):
    '''convert roots to indices'''
    res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
    max_len = max([len(x) for x in res])
    # res = np.array(res, np.int32)
    ans = torch.tensor([pad_seq(line, max_len) for line in res], dtype=torch.int64, device=device)

    return ans


def depth_split(root, depth=0):
    """
        root: Node
        return: dict
    """
    res = defaultdict(list)
    res[depth].append(root)
    for child in root.children:
        for k, v in depth_split(child, depth + 1).items():
            res[k] += v
    return res


def tree2tensor(trees, device=torch.device('cuda')):
    """
        # depthes: n级节点
        # indice: n级节点的子节点集合
        # tree_num: 该节点属于哪棵树
    indice:
        this has structure data.
        0 represent init state,
        1<n represent children's number (1-indexed)
    depthes:
        these are labels of nodes at each depth.
    tree_num:
        explain number of tree that each node was contained.
    """
    res = defaultdict(list)
    tree_num = defaultdict(list)
    for e, root in enumerate(trees):
        for k, v in depth_split(root).items():
            res[k] += v
            tree_num[k] += [e] * len(v)

    for k, v in res.items():
        for e, n in enumerate(v):
            n.num = e + 1
    depthes = [x[1] for x in sorted(res.items(), key=lambda x: -x[0])]
    indices = [get_nums(nodes, device=device) for nodes in depthes]
    depthes = [torch.tensor([n.label for n in nn], dtype=torch.int32, device=device) for nn in depthes]
    tree_num = [
        np.array(x[1], np.int32) for x in sorted(tree_num.items(), key=lambda x: -x[0])]
    return [depthes, indices, tree_num]


def convert_examples_to_features(js, tokenizer, args):
    # code
    code = ' '.join(js['code_tokens'])
    code_tokens = tokenizer.tokenize(code)[:args.code_size]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    json_object = js['ast']
    nodes = [Node(num=i, children=[]) for i in range(len(json_object))]

    for item in json_object:
        idx = item.get('id')
        lable_tokens = tokenizer.tokenize(item.get('label'))[:args.label_size]
        label_ids = tokenizer.convert_tokens_to_ids(lable_tokens)
        padding_length = args.label_size - len(label_ids)
        label_ids += [tokenizer.pad_token_id] * padding_length
        nodes[idx].label = label_ids
        nodes[idx].num = idx
        if item.get('children')[0] != -1:
            children = item.get('children')
            for c in children:
                nodes[idx].children.append(nodes[c])
                nodes[c].parent = nodes[idx]

    tree_root = nodes[0]

    # query
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.query_size]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.query_size - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_ids, tree_root, nl_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, batch_size=32):
        self.examples = []
        self.args = args
        self.batch_size = batch_size
        data = []
        with open(file_path, encoding='utf-8') as f:
            idx = 0
            for line in f:
                # idx=idx+1
                # if(idx>1000):
                #     break
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            d = convert_examples_to_features(js, tokenizer, args)
            self.examples.append(d)

    def __len__(self):
        return len(self.examples) // self.batch_size

    def __getitem__(self, i):
        trees = []
        code_ids = []
        nl_ids = []
        for idx in range(i * self.batch_size, (i + 1) * self.batch_size):
            trees.append(self.examples[idx].tree_root)
            code_ids.append(self.examples[idx].code_ids)
            nl_ids.append(self.examples[idx].nl_ids)

        trees_inputs = tree2tensor(trees, device=self.args.device)

        return (torch.tensor(code_ids),
                trees_inputs,
                torch.tensor(nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.max_steps = args.epoch * (len(train_dataset))
    model.to(args.device)
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)

    best_mrr = 0.0
    model.zero_grad()

    early_stopping_flag = 0
    itr_start_time = time()
    for epoch in range(args.start_epoch, int(args.epoch)):
        if early_stopping_flag > args.early_stopping:
            elapsed = time() - itr_start_time
            t = time_format(elapsed)
            logger.info("Total time:{}".format(t))
            break

        tr_num = 0
        train_loss = 0
        for step in range(len(train_dataset)):
            code_inputs, tree_inputs, nl_inputs = train_dataset[step]
            code_inputs = code_inputs.long().to(args.device)
            nl_inputs = nl_inputs.long().to(args.device)

            model.train()
            loss, code_vec, nl_vec = model(code_inputs, tree_inputs, nl_inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(epoch, step + 1, avg_loss))

        if args.save_every > 0 and epoch % args.save_every == 0:
            results = evaluate(args, model, tokenizer, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))

            early_stopping_flag = early_stopping_flag + 1

            if results['eval_mrr'] > best_mrr:
                early_stopping_flag = 0
                best_mrr = results['eval_mrr']
                logger.info("  " + "*" * 20)
                logger.info("  Best mrr:%s", round(best_mrr, 4))
                logger.info("  " + "*" * 20)

                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)


eval_dataset = None


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, args.eval_batch_size)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs = []
    nl_vecs = []
    for step in range(len(eval_dataset)):
        code_inputs, tree_inputs, nl_inputs = eval_dataset[step]
        code_inputs = code_inputs.to(args.device)
        nl_inputs = nl_inputs.to(args.device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(code_inputs, tree_inputs, nl_inputs)
            eval_loss += lm_loss.mean().item()
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(nl_vecs, code_vecs.T)
    ranks = []
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1 / rank)

    result = {
        "eval_loss": float(perplexity),
        "eval_mrr": float(np.mean(ranks))
    }

    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file, args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset) * args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs = []
    nl_vecs = []

    start = time()
    for step in range(len(eval_dataset)):
        code_inputs, tree_inputs, nl_inputs = eval_dataset[step]
        code_inputs = code_inputs.to(args.device)
        nl_inputs = nl_inputs.to(args.device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(code_inputs, tree_inputs, nl_inputs)
            eval_loss += lm_loss.mean().item()
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1
    code_reprs, doc_reprs = np.vstack(code_vecs), np.vstack(nl_vecs)
    eval_loss = eval_loss / nb_eval_steps
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []

    for i in tqdm(range(0, nb_eval_steps)):
        doc_vec = np.expand_dims(doc_reprs[i], axis=0)
        sims = np.dot(code_reprs, doc_vec.T)[:, 0]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)

        predict_1 = [int(predict[0])]
        predict_5 = [int(k) for k in predict[0:5]]
        predict_10 = [int(k) for k in predict[0:10]]

        sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)

        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1 / float(rank + 1))

    logger.info("***** Test results *****")
    logger.info(
        f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}'
    )
    end = time()
    hour, minute, second = time_format(end - start)
    logger.info("  Testing time: %d h %d m %d s", hour, minute, second)
    logger.info("***** End testing *****")


def main():
    parser = argparse.ArgumentParser()

    ## path parameters
    parser.add_argument("--train_data_file", default='../dataset/jdt/train.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default='../dataset/jdt/valid.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default='../dataset/jdt/test.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tokenizer_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")

    ## model parameters
    parser.add_argument("--code_size", default=300, type=int, )
    parser.add_argument("--label_size", default=5, type=int)
    parser.add_argument("--query_size", default=30, type=int,
                        help="Optional api sequence length after tokenization.")
    parser.add_argument("--vocab_size", default=100000, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--reload", default=False, type=bool,
                        help="Whether to reload.")
    parser.add_argument("--early_stopping", default=3, type=int,
                        help="Whether to reload.")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")

    ## train parameters
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_every', type=int, default=3,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", type=bool, default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=200,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning(" device: %s, n_gpu: %s",
                   device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = Model(tokenizer, args)

    if args.reload == True:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.train_batch_size)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
