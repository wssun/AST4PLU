# introduction
This subdirectory contrains code for BiLSTM and Transformer.

In our study, the input of sequence models can be **Code Tokens**, **BFS**, **SBT**, **SBT w/o Token**, **Token + SBT w/o Token**, **AST Path**.

# Data
For how to process the BigCloneBench dataset into ASTs and preprocess the AST into BFS, SBT, and AST Path, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: data.jsonl, train.txt, valid.txt, test.txt.

## data.jsonl
Filed names of **Code Tokens**, **BFS**, **SBT**, **SBT w/o Token**, **AST Path** are:

Input type|Field name
---|---
Code Tokens|func
BFS|bfs
SBT|sbt
SBT w/o Token|sbt
AST Path|ast_path


# Run
**If the input is AST Path**, run the ``run_path.py`` file in the code directory, 
**else** run the ``run.py`` file in the code directory.

## parameter setting
The explanation of each parameter can be found in the code. Here are several special parameters introduced in detail.

parameter|explanation
---|---
input type|Determine the input type, possible values are: ``func/bfs/sbt/mix``, which correspond to Token/BFS/SBT or SBT w/o Token/Token+(SBT w/o Token)
code size|When input type is ``func/bfs/sbt``, this parameter limit the max length of the input. When input type is ``mix``, this parameter limit the max length of Token.
sbt size|This parameter is only used when input type is ``mix``, and to limit the max length of SBT w/o Token.


# Test
After running ``run.py/run_path.py``, you will get ``predictions.txt`` in the output directory.

You can move ``predictions.txt`` to the evaluator directory and run the following command to get the evaluation result.

```buildoutcfg
cd ./evaluator
python evaluator -a ../dataset/test.txt -p predictions.txt
```