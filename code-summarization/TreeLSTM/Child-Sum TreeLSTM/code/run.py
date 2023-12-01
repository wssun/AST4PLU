# coding=utf-8

from __future__ import absolute_import
import os
import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from collections import defaultdict, Counter
from time import time
from model import Seq2Seq
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
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


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx

            source = js['ast']

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
                 tree,
                 target_ids,
                 target_mask,
    ):
        self.example_id = example_id
        self.tree = tree
        self.target_ids = target_ids
        self.target_mask = target_mask


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        row_tree = example.source
        nodes = [Node(num=i, children=[]) for i in range(len(row_tree))]

        for item in row_tree:
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
                 tree_root,
                 target_ids,
                 target_mask,
            )
        )

    return features



def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return(res)


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


def tree2tensor(trees, device=torch.device('cuda')):
    '''
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



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, batch_size=32, stage=None):
        self.examples = read_examples(file_path)
        self.args = args
        self.batch_size = batch_size
        self.features = convert_examples_to_features(self.examples, tokenizer, args, stage)

    def __len__(self):
        return len(self.examples) // self.batch_size

    def __getitem__(self, i):
        trees = []
        target_ids = []
        target_mask = []
        for idx in range(i * self.batch_size, (i + 1) * self.batch_size):
            trees.append(self.features[idx].tree)
            target_ids.append(self.features[idx].target_ids)
            target_mask.append(self.features[idx].target_mask)

        trees_inputs = tree2tensor(trees, device=self.args.device)

        return (trees_inputs,
                torch.tensor(target_ids, device=self.args.device),
                torch.tensor(target_mask, device=self.args.device))




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
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default="./saved_models/checkpoint-best-bleu/pytorch_model.bin", type=str,
                        help="Path to trained model: Should contain the .bin files" )
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--vocab_size", default=50265, type=int)
    parser.add_argument("--code_size", default=300, type=int)
    parser.add_argument("--label_size", default=5, type=int)
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--reload", default=False, type=bool)
    parser.add_argument("--save_every", default=1, type=int, help="")

    ## Other parameters
    parser.add_argument("--train_filename", default='../dataset/jdt/train.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default='../dataset/jdt/valid.jsonl', type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default='../dataset/jdt/test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")  

    parser.add_argument("--tokenizer_name", default="../../BiLSTM/microsoft/codebert-base", type=str,
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


    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    #budild model
    model=Seq2Seq(args=args,
                  beam_size=args.beam_size, max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.reload:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_dataset = TextDataset(tokenizer, args, args.train_filename, args.train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataset) * args.num_train_epochs
        optimizer = AdamW(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset)*args.train_batch_size)
        logger.info("  Num step = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
        early_stopping_flag = 0
        start = time()
        for epoch in range(args.num_train_epochs):
            tr_num = 0
            train_loss = 0
            for step in range(len(train_dataset)):
                tree_input, target_ids, target_mask = train_dataset[step]
                loss,_,_ = model(tree_input=tree_input,
                                 target_ids=target_ids, target_mask=target_mask)
                # loss, _, _ = model(source_ids=source_ids)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                #Update parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                tr_num += 1
                train_loss += loss.item()
                avg_loss = round(train_loss / tr_num, 5)
                if (step + 1) % 100 == 0:
                    logger.info("epoch {} step {} loss {}".format(epoch, step + 1, avg_loss))

            if args.save_every > 0 and epoch % args.save_every == 0:
                early_stopping_flag += 1
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_data = TextDataset(tokenizer, args, args.dev_filename, args.eval_batch_size, stage="test")
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                model.eval() 
                p=[]
                for step in range(len(eval_data)):
                    tree_input, target_ids, target_mask = eval_data[step]
                    with torch.no_grad():
                        preds = model(tree_input=tree_input)
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w', encoding='utf-8') as f, \
                        open(os.path.join(args.output_dir,"dev.gold"),'w', encoding='utf-8') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  " + "*" * 20)
                logger.info("  Epoch :%s", epoch)
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
        # if args.dev_filename is not None:
        #     files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_data = TextDataset(tokenizer, args, file, args.eval_batch_size, stage="test")

            model.eval() 
            p=[]
            start = time()
            for step in range(len(eval_data)):
                tree_input, target_ids, target_mask = eval_data[step]
                with torch.no_grad():
                    preds = model(tree_input=tree_input)
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w', encoding='utf-8') as f,\
                    open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w', encoding='utf-8') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx))) 
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],5)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            end = time()
            hour, minute, second = time_format(end - start)
            logger.info("  Testing time: %d h %d m %d s", hour, minute, second)
            logger.info("  "+"*"*20)    


if __name__ == "__main__":
    main()


