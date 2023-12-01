# coding=utf-8


from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from time import time
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def time_format(sec):
    hour = sec//3600
    sec = sec % 3600
    minute = sec//60
    second = sec % 60
    return hour, minute, second

class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename, mode):
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx>500:
                break
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx

            if mode == 'token':
                source = ' '.join(js['code_tokens']).replace('\n', ' ')
                source = ' '.join(source.strip().split())
            elif mode == 'sbt':
                source = ' '.join(js['sbt']).replace('\n', ' ')
                source = ' '.join(source.strip().split())
            else:
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())

                sbt = ' '.join(js['sbt']).replace('\n', ' ')
                sbt = ' '.join(sbt.strip().split())

                source = (code, sbt)


            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=source,
                        target = nl,
                        ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 source_ids2=None,
                 source_mask2=None,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.source_ids2 = source_ids2
        self.source_mask2 = source_mask2
        


def convert_examples_to_features(examples, tokenizer, args,stage=None):
    if args.mode == 'token':
        max_source_length = args.code_size
    elif args.mode == 'sbt':
        max_source_length = args.sbt_size

    features = []
    for example_index, example in enumerate(examples):
        #source
        if args.mode == 'token' or args.mode == 'sbt':
            source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
            source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids+=[tokenizer.pad_token_id]*padding_length
            source_mask+=[0]*padding_length

            source_ids2 = [-1]
            source_mask2 = [-1]
        else:
            source_tokens = tokenizer.tokenize(example.source[0])[:args.code_size - 2]
            source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = args.code_size - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length

            source_tokens2 = tokenizer.tokenize(example.source[1])[:args.sbt_size - 2]
            source_tokens2 = [tokenizer.cls_token] + source_tokens2 + [tokenizer.sep_token]
            source_ids2 = tokenizer.convert_tokens_to_ids(source_tokens2)
            source_mask2 = [1] * (len(source_tokens2))
            padding_length = args.sbt_size - len(source_ids2)
            source_ids2 += [tokenizer.pad_token_id] * padding_length
            source_mask2 += [0] * padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 source_ids2,
                 source_mask2
            )
        )
    return features


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
    parser.add_argument("--valid_every", default=3, type=int)
    parser.add_argument("--mode", default='sbt', type=str)
    parser.add_argument("--code_size", default=300, type=int)
    parser.add_argument("--sbt_size", default=1500, type=int) # 1500
    parser.add_argument("--early_stop", default=5, type=int)
    parser.add_argument("--reload", default=False, type=bool)

    ## Other parameters
    parser.add_argument("--train_filename", default='../dataset/jdt/train.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default='../dataset/jdt/valid.jsonl', type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default='../dataset/jdt/test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")  

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 

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
    
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
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

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    #budild model

    model=Seq2Seq(args=args,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.reload:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename, mode=args.mode)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        source_ids2 = torch.tensor([f.source_ids2 for f in train_features], dtype=torch.long)
        source_mask2 = torch.tensor([f.source_mask2 for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
        train_data = TensorDataset(source_ids, source_mask, source_ids2, source_mask2,
                                   all_target_ids, all_target_mask)
        
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
        logger.info("  Num examples = %d", len(train_examples))
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
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,source_ids2,source_mask2,target_ids,target_mask = batch
                loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,
                                 source_ids2=source_ids2, source_mask2=source_mask2,
                                 target_ids=target_ids,target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
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
                    eval_examples = read_examples(args.dev_filename, mode=args.mode)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    source_ids2 = torch.tensor([f.source_ids2 for f in eval_features], dtype=torch.long)
                    source_mask2 = torch.tensor([f.source_mask2 for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(source_ids,source_mask,source_ids2,source_mask2)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,source_ids2,source_mask2= batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask,
                                      source_ids2=source_ids2, source_mask2=source_mask2)
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
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

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
        model = Seq2Seq(args=args,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        model.to(device)
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)

        files=[]
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file, mode=args.mode)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            source_ids2 = torch.tensor([f.source_ids2 for f in eval_features], dtype=torch.long)
            source_mask2 = torch.tensor([f.source_mask2 for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(source_ids,source_mask,source_ids2,source_mask2)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            start = time()
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, source_ids2, source_mask2= batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask,
                                  source_ids2=source_ids2, source_mask2=source_mask2)
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
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

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


