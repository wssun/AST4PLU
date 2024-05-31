# Introduction
This subdirectory contrains code for AST-Trans.

# Data
For how to process the BigCloneBench dataset into ASTs, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: train.jsonl, valid.jsonl, test.jsonl.


# Run
1. Run ``process.py`` in the code directory. The explanation of parameters can be found in the code.
   
   You will get ``un_split_matrices.npz``, ``un_split_pot.jsonl`` and ``un_split_sbt.jsonl`` in the dataset directory.

2. Run ``run.py`` in the code directory. The explanation of parameters can be found in the code.

# Test
After running ``run.py``, you will get ``test_0.output`` and ``test_0.gold`` in the output directory.

You can use them to calculate METEOR and ROUGE-L.
