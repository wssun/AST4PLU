# AST4PLU
**Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?**    
[![arXiv](https://img.shields.io/badge/arXiv-2312.00413-b31b1b.svg)](https://arxiv.org/abs/2312.00413)
```
@article{2023-AST4PLU,
  title={Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?},
  author={Weisong Sun, Chunrong Fang, Yun Miao, Yudu You, Mengzhe Yuan, Yuchen Chen, Quanjun Zhang, An Guo, Xiang Chen, Yang Liu, Zhenyu Chen},
  journal={arXiv preprint arXiv:2312.00413},
  year={2023}
}
```

## Introduction
This repository contains the implementation code for the experiments in the paper "Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?".

**data-process** - Contains code for generating AST using JDT, srcML, Antlr and Tree-sitter, and code for preprocessing AST into BFS sequence, SBT, AST Path, Binary Tree and Split AST.

**code-clone-detection** - Contains code for the code clone detection task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

**code-summarization** - Contains code for the code summarization task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

**code-search** - Contains code for the code search task using BiLSTM, Transformer, Child-Sum TreeLSTM, N-ary TreeLSTM and AST-Trans.

The supplemental material of our paper is ./TOSEM 2024_Supplemental Material.pdf.

## Install environment
pip install -r requirements.txt

pytorch-geometric (used for AST-Trans) can be installed by ``pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.10.0%2Bcu102.html``

## Data preprocess
Please refer to the README in data-process directory.

Dataset preprocessed by us can be downloaded [here](https://drive.google.com/drive/folders/12h4SrBcqW31FsP0faXuJjo0wwO7lOTW0?usp=sharing).

## Run
Run ``run.py`` in the code directory. 

For specific parameter settings, please refer to the README under each subfolder.

## Test
For code clone detection, run ``evaluator.py`` in the evaluator directory. 

For other tasks, please refer to the README under each subfolder.

## Result
Results of our experiments can be found [here](https://drive.google.com/drive/folders/1FrBcqhKpvzfwQ1ajtwqe8xqPfMsr3_w2?usp=sharing).
