# coding=utf-8


from __future__ import absolute_import, division, print_function
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from time import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from tqdm import tqdm, trange
import multiprocessing
from model import Model
import setproctitle
from fast_ast_data_set import FastASTDataSet
from torch_geometric.data import Data

cpu_cont = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)
fh = logging.FileHandler(f'logs.txt')
logger.addHandler(fh)  # add the handlers to the logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def time_format(sec):
    hour = sec//3600
    sec = sec % 3600
    minute = sec//60
    second = sec % 60
    return hour, minute, second


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, device):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    args.max_steps = args.epoch*len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    optimizer = AdamW(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)

    best_mrr = 0.0
    model.zero_grad()
    early_stopping_flag = 0
    start = time()
    for epoch in range(args.start_epoch, int(args.epoch)):
        bar = train_dataloader
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            ast_ids, par_edges, bro_edges, nl_ids, nl_mask = batch
            x = Data(ast_ids=ast_ids,
                     par_edges=par_edges,
                     bro_edges=bro_edges,
                     nl_ids=nl_ids,
                     nl_mask=nl_mask)

            x.to(device)

            model.train()
            loss, code_vec, nl_vec = model(x)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss/tr_num, 5)
            if (step+1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(
                    epoch, step+1, avg_loss))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if args.save_every > 0 and epoch % args.save_every == 0:
            results = evaluate(args, model, device, eval_when_training=True)
            early_stopping_flag += 1
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))

            if results['eval_mrr'] > best_mrr:
                early_stopping_flag = 0
                best_mrr = results['eval_mrr']
                logger.info("  "+"*"*20)
                logger.info("  Best mrr:%s", round(best_mrr, 4))
                logger.info("  "+"*"*20)

                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(
                    args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

            if early_stopping_flag > 3:
                break
    end = time()
    hour, minute, second = time_format(end-start)
    logger.info("  Training time: %d h %d m %d s", hour, minute, second)
    logger.info("***** End training *****")


eval_dataset = None
def evaluate(args, model, device, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        eval_dataset = FastASTDataSet(args, 'test')

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size)

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
    for batch in eval_dataloader:
        ast_ids, par_edges, bro_edges, nl_ids, nl_mask = batch
        x = Data(ast_ids=ast_ids,
                 par_edges=par_edges,
                 bro_edges=bro_edges,
                 nl_ids=nl_ids,
                 nl_mask=nl_mask)

        x.to(device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(x)
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
        ranks.append(1/rank)

    result = {
        "eval_loss": float(perplexity),
        "eval_mrr": float(np.mean(ranks))
    }

    return result


def test(args, model, device):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = FastASTDataSet(args, 'test')

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
    code_vecs = []
    nl_vecs = []
    start = time()
    for batch in eval_dataloader:
        ast_ids, par_edges, bro_edges, nl_ids, nl_mask = batch
        x = Data(ast_ids=ast_ids,
                 par_edges=par_edges,
                 bro_edges=bro_edges,
                 nl_ids=nl_ids,
                 nl_mask=nl_mask)

        x.to(device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(x)
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
    hour, minute, second = time_format(end-start)
    logger.info("  Testing time: %d h %d m %d s", hour, minute, second)
    logger.info("***** End testing *****")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='../dataset', type=str)
    parser.add_argument("--output_dir", default='saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--reload", default=False, type=bool,
                        help="Whether to reload.")
    parser.add_argument("--vocab_size", default=60000, type=int)
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)

    # Other parameters
    parser.add_argument("--tokenizer_name_or_path", default='../microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--query_size", default=30, type=int,
                        help="Optional api sequence length after tokenization.")
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_research", default=False, type=bool,
                        help="Whether to run research on the dev set.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_every', type=int, default=1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=200,
                        help="random seed for initialization")

    # ast-trans
    parser.add_argument("--data_type", default='sbt', type=str, help="pot/sbt, corresponding to sequence data format")
    parser.add_argument("--max_src_len", default=1500, type=int,help="length of pot/sbt")  # 1500
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

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",
                   device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer

    args.start_epoch = 0
    args.start_step = 0

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)

    model = Model(
        args=args,
        hidden_size=args.hidden_size,
        par_heads=args.par_heads, num_heads=args.num_heads,
        max_rel_pos=args.max_rel_pos,
        pos_type=args.pos_type,
        num_layers=args.num_layers,
        dim_feed_forward=args.dim_feed_forward,
        dropout=args.dropout)

    if args.reload:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = FastASTDataSet(args, 'train')
        train(args, train_dataset, model, device)

    # Evaluation
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, device)


if __name__ == "__main__":
    setproctitle.setproctitle("YYD")
    main()

