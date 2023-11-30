# AST4PLU
Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?

## Introduction
This repository contains the implementation code for the experiments in the paper "Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?".

**data-process** - Contains code for generating AST using JDT, srcML, Antlr and Tree-sitter, and code for preprocessing AST into BFS sequence, SBT, AST Path, Binary Tree and Split AST.

**code-clone-detection** - Contains code for the code clone detection task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

**code-summarization** - Contains code for the code summarization task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

**code-search** - Contains code for the code search task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

## Install environment
pip install -r requirements.txt

pytorch-geometric can be installed by pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html

## Data preprocess
Please refer to the README in data process directory.

## Run
Run run.py in the code directory. 

For specific parameter settings, please refer to the README under each subfolder.

## Test
Run evaluator.py in the evaluator directory. 
