# Introduction
This directory contains files to generate AST and preprocess AST.

**AST Parsers** - We use JDT, SrcML, Antlr and Tree-sitter in our experiment. You can find a simple guide about how to use these parsers below.

**AST Preprocessing** - We provide the code to process AST into BFS, SBT, SBT w/o Token, AST Path, Binary Tree and Split AST in the ways decscribed in our paper.

The table below shows the combination of AST parsers and preprocessing methods used in our experiment.

 |AST|BFS|SBT|SBT w/o Token|AST Path|Binary Tree|Split AST
---|---|---|---|---|---|---|---
JDT|√|√|√|√|√|√|-
SrcML|√|-|√|-|-|-|-
Antlr|√|-|√|-|-|-|-
Tree-sitter|√|-|√|-|-|-|√

# Install AST parsers
The steps for installing JDT, SrcML, Antlr and Tree-sitter can be easily found online.
We recorded our installation steps, problems we encountered and how we fixed them in ``JDT.pdf``, ``SrcML.pdf``, ``Antlr.pdf`` and ``Tree-sitter.pdf`` as a reference to you. (These pdfs are written in Chinese.)

# Generate and Process ASTs

## JDT
JDT can only be used in Eclipse.

### Original AST, BFS, SBT
The code for using JDT to generate the AST for a single function is in ``data-process/process/src/main/java/JDT/GenerateAST.java``.
The code for processing BigCloneBench and CodeSearchNet is in ``data-process/process/src/main/java/process/JDT_for_bcb.java`` and ``data-process/process/src/main/java/process/JDT_for_csn.java`` respectively.

``FILE_PATH`` is path to original data file (a json file) in BigCloneBench or CodeSearchNet.
``JSON_FILE_PATH`` is path to the result data file (a json file).

### SBT w/o Token

### AST Path

### Binary Tree


