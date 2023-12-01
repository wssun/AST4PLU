# coding=utf-8

from __future__ import absolute_import
import os
import bleu
import torch
import random
import logging
import argparse
import numpy as np
from io import open
from time import time
from fast_ast_data_set import FastASTDataSet
from model import FastASTTrans
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from torch_geometric.data import Data
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default='../microsoft/codebert-base', type=str,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default='./saved_models/checkpoint-best-bleu/pytorch_model.bin', type=str,
                        help="Path to trained model: Should contain the .bin files" )
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--vocab_size", default=50265, type=int)
    parser.add_argument("--valid_every", default=1, type=int)
    parser.add_argument("--mode", default='sbt', type=str)
    parser.add_argument("--early_stop", default=5, type=int)
    parser.add_argument("--reload", default=False, type=bool)

    ## Other parameters
    parser.add_argument("--data_dir", default='../dataset', type=str)
    parser.add_argument("--data_type", default='sbt', type=str, help="pot/sbt, corresponding to sequence data format")


    parser.add_argument("--max_target_length", default=30, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    # ast-trans
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

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)


    model = FastASTTrans(
                    args=args,
                    beam_size=args.beam_size,
                    max_target_length=args.max_target_length,
                    tokenizer=tokenizer,
                    vocab_size=args.vocab_size,
                    hidden_size=args.hidden_size,
                    par_heads=args.par_heads, num_heads=args.num_heads,
                    max_rel_pos=args.max_rel_pos,
                    pos_type=args.pos_type,
                    num_layers=args.num_layers,
                    dim_feed_forward=args.dim_feed_forward,
                    dropout=args.dropout)

    if args.reload:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_data = FastASTDataSet(args, 'train')

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
        early_stopping_flag = 0
        start = time()
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                source_ids, par_edges, bro_edges, target_ids, target_mask = batch
                x = Data(source_ids=source_ids,
                         par_edges=par_edges,
                         bro_edges=bro_edges,
                         target_ids=target_ids,
                         target_mask=target_mask)

                x.to(device)
                loss,_,_ = model(x)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.valid_every > 0 and epoch % args.valid_every == 0:
                early_stopping_flag += 1
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_data = FastASTDataSet(args, 'test')
                    eval_examples = eval_data.ast_data
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    source_ids, par_edges, bro_edges, target_ids, target_mask = batch
                    x = Data(source_ids=source_ids,
                             par_edges=par_edges,
                             bro_edges=bro_edges,
                             target_ids=target_ids,
                             target_mask=target_mask)

                    x.to(device)
                    with torch.no_grad():
                        preds = model(x)

                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w', encoding='utf-8') as f, open(os.path.join(args.output_dir,"dev.gold"),'w',encoding='utf-8') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.nl+'\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  " + "*" * 20)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    early_stopping_flag = 0
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

            if early_stopping_flag > args.early_stop:
                break
        end = time()
        hour, minute, second = time_format(end - start)
        logger.info("  Training time: %d h %d m %d s", hour, minute, second)
        logger.info("***** End training *****")
               
    if args.do_test:
        model = FastASTTrans(
                        args=args,
                        beam_size=args.beam_size,
                        max_target_length=args.max_target_length,
                        tokenizer=tokenizer,
                        vocab_size=args.vocab_size,
                        hidden_size=args.hidden_size,
                        par_heads=args.par_heads, num_heads=args.num_heads,
                        max_rel_pos=args.max_rel_pos,
                        pos_type=args.pos_type,
                        num_layers=args.num_layers,
                        dim_feed_forward=args.dim_feed_forward,
                        dropout=args.dropout)

        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        model.to(device)
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)

        files = ['test']

        for idx,file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_data = FastASTDataSet(args, file)
            eval_examples = eval_data.ast_data

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p=[]
            start = time()
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                source_ids, par_edges, bro_edges, target_ids, target_mask = batch
                x = Data(source_ids=source_ids,
                         par_edges=par_edges,
                         bro_edges=bro_edges,
                         target_ids=target_ids,
                         target_mask=target_mask)

                x.to(device)
                with torch.no_grad():
                    preds = model(x)
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w',encoding='utf-8') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w',encoding='utf-8') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.nl+'\n')

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx)))

            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)

            end = time()
            hour, minute, second = time_format(end - start)
            logger.info("  Testing time: %d h %d m %d s", hour, minute, second)
            logger.info("***** End testing *****")


                
if __name__ == "__main__":
    main()


