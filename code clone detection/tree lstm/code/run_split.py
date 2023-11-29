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
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
from collections import Counter, defaultdict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import setproctitle
from transformers import RobertaTokenizer


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model_split import Model
from sklearn.metrics import recall_score, precision_score, f1_score

cpu_cont = 16
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

UNK_ID = 3
PAD_ID = 1


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 tree1,
                 tree2,
                 label,
                 url1,
                 url2
                 ):
        self.tree1 = tree1
        self.tree2 = tree2
        self.label = label
        self.url1 = url1
        self.url2 = url2


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def build_example(asts, tokenizer, args):
    final_nodes = []
    for ast in asts:
        nodes = [Node(num=i, children=[]) for i in range(len(ast))]
        for item in ast:
            idx = item.get('id')
            label_tokens = tokenizer.tokenize(item.get('label'))[:args.label_size]
            label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            padding_length = args.label_size - len(label_ids)
            label_ids += [tokenizer.pad_token_id] * padding_length
            nodes[idx].label = label_ids
            nodes[idx].num = idx
            if item.get('children')[0] != -1:
                children = item.get('children')
                for c in children:
                    nodes[idx].children.append(nodes[c])
                    nodes[c].parent = nodes[idx]
        final_nodes.append(nodes[0])

    return final_nodes   # roots of split asts


def sample_trees(trees, random_sample):
    pad_ast=[{"children":[-1],"id":0,"label":"<pad>"}]
    if len(trees) < random_sample:
        while len(trees) < random_sample:
            trees.append(pad_ast)
        return trees
    else:
        index = random.sample(range(0, len(trees)), random_sample)
        final_tree = []
        for i in index:
            final_tree.append(trees[i])
        return final_tree


class TextDataset(Dataset):
    def __init__(self, args, file_path, tokenizer=None, batch_size=32):
        self.examples = []
        self.batch_size = batch_size
        self.args = args
        self.tokenizer = tokenizer
        logger.info("Creating features from index file at %s ", file_path)
        url_to_code = {}
        idx = 0
        with open(args.data_file, encoding='utf-8') as f:
            for line in f:
                # idx = idx + 1  # debugging
                # if idx > 200:  # debugging
                #     break  # debugging
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = sample_trees(js['asts'], args.sample_trees)

        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label == '0':
                    label = 0
                else:
                    label = 1

                data.append((url1, url2, label))

        for item in tqdm(data):
            url1, url2, label = item
            tree1 = url_to_code[url1]
            tree2 = url_to_code[url2]
            self.examples.append(InputFeatures(tree1, tree2, label, url1, url2))


    def __len__(self):
        return len(self.examples)//self.batch_size

    def __getitem__(self, i):
        trees_inputs1=[]
        trees_inputs2=[]
        labels=[]
        for idx in range(i*self.batch_size, (i+1)*self.batch_size):
            trees_inputs1.extend(build_example(self.examples[idx].tree1, self.tokenizer, self.args))
            trees_inputs2.extend(build_example(self.examples[idx].tree2, self.tokenizer, self.args))
            labels.append(self.examples[idx].label)

        input1 = tree2tensor(trees_inputs1,self.args.device)
        input2 = tree2tensor(trees_inputs2,self.args.device)

        return input1, input2, torch.tensor(labels)


def tree2tensor(trees, device=torch.device('cuda')):
    '''
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
    '''
    res = defaultdict(list)
    tree_num = defaultdict(list)
    for e, root in enumerate(trees):
        for k, v in depth_split(root).items():
            res[k] += v
            tree_num[k] += [e] * len(v)

    for k, v in res.items():
        for e, n in enumerate(v):
            n.num = e + 1
    depthes = [x[1] for x in sorted(res.items(), key=lambda x:-x[0])]
    indices = [get_nums(nodes, device=device) for nodes in depthes]
    depthes = [torch.tensor([n.label for n in nn], dtype=torch.int32, device=device) for nn in depthes]
    tree_num = [
        np.array(x[1], np.int32) for x in sorted(tree_num.items(), key=lambda x: -x[0])]
    return [depthes, indices, tree_num]


def get_nums(roots, device=torch.device('cuda')):
    '''convert roots to indices'''
    res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
    max_len = max([len(x) for x in res])
    # res = np.array(res, np.int32)
    ans = torch.tensor([pad_seq(line, max_len) for line in res], dtype=torch.int64, device=device)

    return ans


def depth_split(root, depth=0):
    '''
    root: Node
    return: dict
    '''
    res = defaultdict(list)
    res[depth].append(root)
    for child in root.children:
        for k, v in depth_split(child, depth + 1).items():
            res[k] += v
    return res


def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        seq = seq + [-1] * (maxlen - len(seq))

    return seq


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, eval_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.max_steps = args.epoch * (len(train_dataset))

    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    model.zero_grad()

    if args.reload:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch']
        logger.info("Loading checkpoint from %s", checkpoint_path)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset)*args.train_batch_size)
    logger.info("  Start Epoch = %d", args.start_epoch + 1)
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size)

    best_f1 = 0
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    early_stopping_flag = 0
    start_time = time.time()
    for epoch in range(args.start_epoch + 1, args.epoch):
        bar = tqdm(train_dataset, total=len(train_dataset))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            input1, input2, labels = batch
            labels = labels.long().to(args.device)
            model.train()
            loss, logits = model(input1, input2, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(epoch, avg_loss))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if args.save_every > 0 and epoch % args.save_every == 0:
            results = evaluate(args, model, eval_dataset, eval_when_training=True)
            early_stopping_flag += 1

            if results['eval_f1'] > best_f1:
                early_stopping_flag = 0
                best_f1 = results['eval_f1']
                logger.info("  " + "*" * 20)
                logger.info("  Best f1:%s", round(best_f1, 4))
                logger.info("  " + "*" * 20)

                model_to_save = model.module if hasattr(model, 'module') else model
                output_dir = os.path.join(args.output_dir, '{}'.format('checkpoint-best-f1.bin'))
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving best model to %s", output_dir)

            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saving model checkpoint to %s, epoch = %d", checkpoint_path, epoch)

            if early_stopping_flag > 3:
                logger.info("Early stopping, total epoch: %d ", epoch)
                break

    end_time = time.time()
    cost_time = end_time - start_time
    ss = cost_time % 60
    cost_time = cost_time // 60
    mm = cost_time % 60
    cost_time = cost_time // 60
    hh = cost_time
    logger.info("  Training time: %d h %d m %d s", hh, mm, ss)
    logger.info("***** End training *****")


def evaluate(args, model, eval_dataset, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset)*args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataset:
        input1, input2, labels = batch
        labels = labels.long().to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(input1, input2, labels)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0
    best_f1 = 0
    for i in range(1, 100):
        threshold = i / 100
        y_preds = logits[:, 1] > threshold
        f1 = f1_score(y_trues, y_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, eval_dataset, best_threshold=0.57):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset)*args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = time.time()

    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataset:
        input1, input2, labels = batch
        labels = labels.long().to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(input1, input2, labels)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = logits[:, 1] > best_threshold

    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, y_preds):
            if pred:
                f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
            else:
                f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    end_time = time.time()
    cost_time = end_time - start_time
    ss = cost_time % 60
    cost_time = cost_time // 60
    mm = cost_time % 60
    cost_time = cost_time // 60
    hh = cost_time
    logger.info("  Testing time: %d h %d m %d s", hh, mm, ss)
    logger.info("***** End testing *****")


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_file", default="../dataset/data.jsonl", type=str,
                        help="The input code/ast/sbt data file (a json file).")
    parser.add_argument("--train_data_file", default="../dataset/train.txt", type=str,
                        help="The input training data file (a txt file).")
    parser.add_argument("--eval_data_file", default="../dataset/valid.txt", type=str)
    parser.add_argument("--test_data_file", default="../dataset/test.txt", type=str)

    parser.add_argument("--output_dir", default="./split_ast", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--word_vocab_size", default=60000, type=int, help="vocab size of embedding")
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--emb_size", default=512, type=int, help='to reload checkpoint, not used')
    parser.add_argument("--label_size", default=5, type=int, help="number of words in label")
    parser.add_argument("--sample_trees", default=15, type=int, help="randomly sample k trees in every split_ast")
    parser.add_argument("--reload", default=False, type=bool, help="Continue training from checkpoint.")    # debug

    parser.add_argument('--epoch', type=int, default=200)            # debug
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")      # debug
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")       # debug
    parser.add_argument('--save_every', type=int, default=1,
                        help="Save checkpoint every X updates steps.")       # debug

    parser.add_argument("--do_train", action="store_false",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_false",
                        help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action="store_false",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. ")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--tokenizer_name_or_path", default='microsoft/codebert-base', type=str)  # microsoft/codebert-base     ../dataset/codebert-base

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_file_path = os.path.join(args.output_dir, 'log.txt')
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)  # add the handlers to the logger
    logger.info("\n\n")

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",
                   device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = -1
    model = Model(args)
    logger.info("Training/evaluation parameters %s", args)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Training
    if args.do_train:
        train_dataset = TextDataset(args, file_path=args.train_data_file, tokenizer=tokenizer, batch_size=args.train_batch_size)
        eval_dataset = TextDataset(args, file_path=args.eval_data_file, tokenizer=tokenizer, batch_size=args.eval_batch_size)
        train(args, train_dataset, eval_dataset, model)

    # Evaluation
    if args.do_eval or args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)

        eval_dataset = TextDataset(args, file_path=args.eval_data_file, tokenizer=tokenizer, batch_size=args.eval_batch_size)
        result = evaluate(args, model, eval_dataset)

        if args.do_test:
            test_dataset = TextDataset(args, file_path=args.test_data_file, tokenizer=tokenizer, batch_size=args.eval_batch_size)
            test(args, model, test_dataset, best_threshold=result['eval_threshold'])


if __name__ == "__main__":
    proc_title = "clone_detection"
    setproctitle.setproctitle(proc_title)

    print(torch.cuda.current_device())
    main()

