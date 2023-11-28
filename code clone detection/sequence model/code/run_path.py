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
from collections import Counter
import time
import setproctitle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model_path import BiLSTM
from sklearn.metrics import recall_score, precision_score, f1_score

cpu_cont = 16
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def tokenize_and_pad(x, tokenizer, length):
    x = tokenizer.tokenize(x)
    x = x[:length]
    x_ids = tokenizer.convert_tokens_to_ids(x)
    padding_length = length - len(x_ids)
    x_ids += [tokenizer.pad_token_id] * padding_length
    return x_ids


def sample_and_tokenize_paths(paths, tokenizer, args, random_sample):
    left = []
    p = []
    right = []
    for path in paths:
        token1, path, token2 = path.split('<sep>')
        token1 = tokenize_and_pad(token1, tokenizer, args.token_size)
        path = tokenize_and_pad(path, tokenizer, args.path_size)
        token2 = tokenize_and_pad(token2, tokenizer, args.token_size)
        left.append(token1)
        p.append(path)
        right.append(token2)
    if len(left) < args.sample_contexts:
        token_pad = [tokenizer.pad_token_id] * args.token_size
        path_pad = [tokenizer.pad_token_id] * args.path_size
        while len(left) < args.sample_contexts:
            left.append(token_pad)
            p.append(path_pad)
            right.append(token_pad)
    Left = []
    Path = []
    Right = []
    if random_sample:
        index = random.sample(range(0, len(left)), args.sample_contexts)
        for i in index:
            Left.append(left[i])
            Path.append(p[i])
            Right.append(right[i])
    else:
        Left = left[:args.sample_contexts]  # sample_contexts * token_size
        Path = p[:args.sample_contexts]  # sample_contexts * path_size
        Right = right[:args.sample_contexts] # sample_contexts * token_size
    return (Left, Path, Right)


def get_example(item):
    url1, url2, label, tokenizer, args, url_to_code = item
    left1, p1, right1 = url_to_code[url1]
    left2, p2, right2 = url_to_code[url2]
    return InputFeatures(left1, p1, right1, left2, p2, right2, label, url1, url2)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, left1, p1, right1, left2, p2, right2, label, url1, url2):
        self.left1 = left1
        self.p1 = p1
        self.right1 = right1
        self.left2 = left2
        self.p2 = p2
        self.right2 = right2
        self.label = label
        self.url1 = url1
        self.url2 = url2


class TextDataset(Dataset):
    def __init__(self, args, file_path, tokenizer=None, random_sample=False):
        self.examples = []
        logger.info("Creating features from index file at %s ", file_path)
        url_to_code = {}
        idx = 0
        with open(args.data_file, encoding='utf-8') as f:
            for line in f:
                # idx = idx + 1  # debugging
                # if idx > 100:  # debugging
                #     break  # debugging
                line = line.strip()
                js = json.loads(line)
                ast_path = js['ast_path']
                url_to_code[js['idx']] = sample_and_tokenize_paths(ast_path, tokenizer, args, random_sample)

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
                data.append((url1, url2, label, tokenizer, args, url_to_code))

            for item in tqdm(data):
                self.examples.append(get_example(item))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return item

    def get_item(self, item):
        return self.examples[item].left1, self.examples[item].p1, self.examples[item].right1, \
               self.examples[item].left2, self.examples[item].p2, self.examples[item].right2, \
               self.examples[item].label
        # sample_contexts * token_size/path_size/token_size


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def idx_to_data(indices, dataset):
    l1 = []  # batch * sample_contexts * token_size
    p1 = []  # batch * sample_contexts * path_size
    r1 = []  # batch * sample_contexts * token_size
    l2 = []
    p2 = []
    r2 = []
    labels = []
    for idx in indices:
        left1, path1, right1, left2, path2, right2, label = dataset.get_item(idx)
        l1.append(torch.tensor(left1))
        p1.append(torch.tensor(path1))
        r1.append(torch.tensor(right1))
        l2.append(torch.tensor(left2))
        p2.append(torch.tensor(path2))
        r2.append(torch.tensor(right2))
        labels.append(label)

    l1 = torch.stack(l1)
    p1 = torch.stack(p1)
    r1 = torch.stack(r1)
    l2 = torch.stack(l2)
    p2 = torch.stack(p2)
    r2 = torch.stack(r2)
    labels = torch.tensor(labels)

    return l1, p1, r1, l2, p2, r2, labels


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epoch * len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    model.zero_grad()
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

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

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Start Epoch = %d", args.start_epoch + 1)
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    best_f1 = 0
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    early_stopping_flag = 0
    start_time = time.time()
    for epoch in range(args.start_epoch + 1, args.epoch):
        if epoch != args.start_epoch + 1:  # resample k ast paths
            train_dataset = TextDataset(args, file_path=args.train_data_file, tokenizer=tokenizer, random_sample=True)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            left1, p1, right1, left2, p2, right2, label = idx_to_data(batch, train_dataset)
            left1 = left1.long().to(args.device)
            p1 = p1.long().to(args.device)
            right1 = right1.long().to(args.device)
            left2 = left2.long().to(args.device)
            p2 = p2.long().to(args.device)
            right2 = right2.long().to(args.device)
            label = label.long().to(args.device)
            model.train()
            loss, logits = model(left1, p1, right1, left2, p2, right2, label)

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
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1.bin')
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving best model to %s", output_dir)
                logger.info(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saving model checkpoint to %s, epoch = %d", checkpoint_path, epoch)
            logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            if early_stopping_flag > args.early_stop:
                logger.info("Early stopping, total epoch: %d ", epoch)
                logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
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

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        left1, p1, right1, left2, p2, right2, label = idx_to_data(batch, eval_dataset)
        left1 = left1.long().to(args.device)
        p1 = p1.long().to(args.device)
        right1 = right1.long().to(args.device)
        left2 = left2.long().to(args.device)
        p2 = p2.long().to(args.device)
        right2 = right2.long().to(args.device)
        label = label.long().to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(left1, p1, right1, left2, p2, right2, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1
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


def test(args, model, eval_dataset, best_threshold=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    start_time = time.time()
    for batch in eval_dataloader:
        left1, p1, right1, left2, p2, right2, label = idx_to_data(batch, eval_dataset)
        left1 = left1.long().to(args.device)
        p1 = p1.long().to(args.device)
        right1 = right1.long().to(args.device)
        left2 = left2.long().to(args.device)
        p2 = p2.long().to(args.device)
        right2 = right2.long().to(args.device)
        label = label.long().to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(left1, p1, right1, left2, p2, right2, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_preds = logits[:, 1] > best_threshold
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, y_preds):
            if pred:
                f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
            else:
                f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

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
    parser.add_argument("--output_dir", default="ast_path", type=str,
                        help="The output directory where logs and checkpoints will be written.")

    parser.add_argument("--model", default="bilstm", type=str, help="bilstm")
    parser.add_argument('--epoch', type=int, default=200)            # debug
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--token_size", default=5, type=int,help="length of tokens")
    parser.add_argument("--path_size", default=50, type=int, help="length of paths")
    parser.add_argument("--sample_contexts", default=700, type=int, help="randomly sample k paths in every batch")
    parser.add_argument("--vocab_size", default=60000, type=int, help="vocab size of embedding")
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--reload", default=False, type=bool,
                        help="Continue training from checkpoint.")    # debug

    parser.add_argument("--early_stop", default=3, type=int, help="early stop epoch number")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")      # debug
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")       # debug
    parser.add_argument('--save_every', type=int, default=1,
                        help="Save checkpoint every X updates steps.")       # debug

    parser.add_argument("--do_train", action="store_false",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_false",
                        help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action="store_false",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
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
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")  # microsoft/codebert-base  |  ../dataset/codebert-base

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.model)
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

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    args.start_epoch = -1
    # if args.model == 'bilstm':
    model = BiLSTM(args, tokenizer)
    # else:
    #     model = Transformer(args)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(args, file_path=args.train_data_file, tokenizer=tokenizer, random_sample=True)
        eval_dataset = TextDataset(args, file_path=args.eval_data_file, tokenizer=tokenizer, random_sample=False)
        train(args, train_dataset, eval_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval or args.do_test:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)

        eval_dataset = TextDataset(args, file_path=args.eval_data_file, tokenizer=tokenizer, random_sample=False)
        result = evaluate(args, model, eval_dataset)

        if args.do_test:
            test_dataset = TextDataset(args, file_path=args.test_data_file, tokenizer=tokenizer, random_sample=False)
            test(args, model, test_dataset, best_threshold=result['eval_threshold'])


if __name__ == "__main__":
    proc_title = "clone_detection"
    setproctitle.setproctitle(proc_title)

    print(torch.cuda.current_device())
    main()

