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
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from fast_ast_data_set import FastASTDataSet
from torch_geometric.data import Data

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import FastASTTrans
from sklearn.metrics import recall_score, precision_score, f1_score
from ignite.utils import setup_logger, convert_tensor
from transformers import AdamW, get_linear_schedule_with_warmup

cpu_cont = 16
logger = logging.getLogger(__name__)


def _graph_prepare_batch(batch, device=None, non_blocking: bool = False):
    x, y = batch
    return (
        x.to(device),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


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
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            ast1, par1, bro1, ast2, par2, bro2, idx1, idx2, label, y = batch
            x = Data(ast1=ast1, par1=par1, bro1=bro1,
                 ast2=ast2, par2=par2, bro2=bro2,
                 idx1=idx1, idx2=idx2, label=label)
            x, y = _graph_prepare_batch((x,y), args.device)
            model.train()
            y_pred = model(x)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(y_pred, y)
            if tr_num%1000 == 0:
                logger.info("epoch {} tr_num {} loss {}".format(epoch, tr_num, loss))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(epoch, avg_loss))
            # logger.info("epoch {} loss {}".format(epoch, avg_loss))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if args.save_every > 0 and epoch % args.save_every == 0:
            results = evaluate(args, model, eval_dataloader, eval_when_training=True)
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


def evaluate(args, model, eval_dataloader, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        ast1, par1, bro1, ast2, par2, bro2, idx1, idx2, label, y = batch
        x = Data(ast1=ast1, par1=par1, bro1=bro1,
                 ast2=ast2, par2=par2, bro2=bro2,
                 idx1=idx1, idx2=idx2, label=label)
        x, y = _graph_prepare_batch((x, y), args.device)
        with torch.no_grad():
            logit = model(x)
            logits.append(logit.cpu().numpy())
            y_trues.append(y.cpu().numpy())
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
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    start_time = time.time()
    for batch in eval_dataloader:
        ast1, par1, bro1, ast2, par2, bro2, idx1, idx2, label, y = batch
        x = Data(ast1=ast1, par1=par1, bro1=bro1,
                 ast2=ast2, par2=par2, bro2=bro2,
                 idx1=idx1, idx2=idx2, label=label)
        x, y = _graph_prepare_batch((x, y), args.device)
        with torch.no_grad():
            logit = model(x)
            logits.append(logit.cpu().numpy())
            y_trues.append(y.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_preds = logits[:, 1] > best_threshold
    with open(os.path.join(args.output_res_dir, "predictions.txt"), 'w') as f:
        for data, pred in zip(eval_dataset.final_dataset, y_preds):
            if pred:
                f.write(data.idx1 + '\t' + data.idx2 + '\t' + '1' + '\n')
            else:
                f.write(data.idx1 + '\t' + data.idx2 + '\t' + '0' + '\n')

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
    # python run.py --data_dir ../dataset/  --model_name_or_path  ../../clone-detection/dataset/codebert-base  --train_batch_size 8 --eval_batch_size 8
    # CUDA_VISIBLE_DEVICES=0 python run.py --train_batch_size 16 --eval_batch_size 16 --epoch 2
    # CUDA_VISIBLE_DEVICES=0 python run.py --data_dir ../dataset/  --model_name_or_path  ../../clone-detection/dataset/codebert-base  --train_batch_size 2  --eval_batch_size 2
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--data_file", default="../dataset/data.jsonl", type=str,
    #                     help="The input code/ast/sbt data file (a json file).")
    # parser.add_argument("--train_data_file", default="../dataset/train.txt", type=str,
    #                     help="The input training data file (a txt file).")
    # parser.add_argument("--eval_data_file", default="../dataset/valid.txt", type=str)
    # parser.add_argument("--test_data_file", default="../dataset/test.txt", type=str)

    parser.add_argument("--data_dir", default='../dataset/', type=str)   # 'D:\\ast_dataset\\bcb\\ast\jdt\\'
    parser.add_argument("--output_dir", default="ast-trans", type=str,
                        help="The output directory where logs and checkpoints will be written.")
    parser.add_argument("--output_res_dir", default="ast-trans", type=str,
                        help="Where model predictions will be written.")

    parser.add_argument('--epoch', type=int, default=200)            # debug
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--vocab_size", default=60000, type=int, help="vocab size of embedding")
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--reload", default=False, type=bool,
                        help="Continue training from checkpoint.")    # debug

    parser.add_argument("--data_type", default='sbt', type=str, help="pot/sbt, corresponding to sequence data format")
    parser.add_argument("--early_stop", default=3, type=int, help="early stop epoch number")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")      # debug
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")       # debug
    parser.add_argument('--save_every', type=int, default=1,
                        help="Save checkpoint every X updates steps.")       # debug

    parser.add_argument("--do_train", action="store_false",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_false",
                        help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action="store_false",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
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
    parser.add_argument("--model_name_or_path", default="../dataset/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")  # microsoft/codebert-base  |  ../dataset/codebert-base

    # ast-trans
    parser.add_argument("--max_src_len", default=1500, type=int,help="length of pot/sbt")
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--par_heads', type=int, default=1)
    parser.add_argument('--max_rel_pos', type=int, default=7)
    parser.add_argument('--max_par_rel_pos', type=int, default=7)
    parser.add_argument('--max_bro_rel_pos', type=int, default=7)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feed_forward', type=int, default=2048)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--is_ignore', type=bool, default=True)
    parser.add_argument('--pos_type', type=str, default='p2q_p2k_p2v')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_res_dir):
        os.makedirs(args.output_res_dir)
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
    model = FastASTTrans(args.vocab_size,
                         args.hidden_size,
                         args.par_heads, args.num_heads,
                         args.max_rel_pos,
                         args.pos_type,
                         args.num_layers,
                         args.dim_feed_forward,
                         args.dropout)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = FastASTDataSet(args, 'train')
        eval_dataset = FastASTDataSet(args, 'valid')
        train(args, train_dataset, eval_dataset, model)

    # Evaluation
    if args.do_eval or args.do_test:
        output_path = os.path.join(args.output_dir, 'checkpoint-best-f1.bin')
        model.load_state_dict(torch.load(output_path))
        model.to(args.device)

        eval_dataset = FastASTDataSet(args, 'valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        result = evaluate(args, model, eval_dataloader)

        if args.do_test:
            test_dataset = FastASTDataSet(args, 'test')
            test(args, model, test_dataset, best_threshold=result['eval_threshold'])


if __name__ == "__main__":
    proc_title = "clone_detection"
    setproctitle.setproctitle(proc_title)

    main()

