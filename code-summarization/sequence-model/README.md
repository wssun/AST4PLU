# Introduction
This subdirectory contrains code for BiLSTM and Transformer.

In our study, the input of sequence models can be **Code Tokens**, **BFS**, **SBT**, **SBT w/o Token**, **Token + SBT w/o Token**, **AST Path**.

# Data
For how to process the BigCloneBench dataset into ASTs and preprocess the AST into BFS, SBT, SBT w/o Token, and AST Path, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: train.jsonl, valid.jsonl, test.jsonl.


# Run
run the ``run.py`` file in the code directory.

## parameter setting
The explanation of each parameter can be found in the code. Here are several special parameters introduced in detail.

| parameter | explanation                                                                                                                            |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------|
| mode      | Determine the input type, possible values are: ``token/sbt/mix``, which correspond to Token, (SBT w/o Token) and Token+(SBT w/o Token) |
| code size | This parameter limits the max length of Token.                                                                                         |
| sbt size  | This parameter limits the max length of SBT w/o Token.                                                                                 |

# Test
After running ``run.py``, you will get ``test_0.output`` and ``test_0.gold`` in the output directory.

You can use them to calculate METEOR and ROUGE-L.

