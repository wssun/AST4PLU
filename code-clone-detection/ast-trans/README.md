# Introduction
This subdirectory contrains code for AST-Trans.

# Data
For how to process the BigCloneBench dataset into ASTs, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: data.jsonl, train.txt, valid.txt, test.txt.


# Run
1. Run ``process.py`` in the code directory. The explanation of parameters can be found in the code.
   
   You will get ``un_split_matrices.npz``, ``un_split_pot.jsonl`` and ``un_split_sbt.jsonl`` in the dataset directory.

2. Run ``run.py`` in the code directory. The explanation of parameters can be found in the code.

# Test
After running ``run.py``, you will get ``predictions.txt`` in the output directory.

You can move ``predictions.txt`` to the evaluator directory and run the following command to get the evaluation result.

```buildoutcfg
cd ./evaluator
python evaluator -a ../dataset/test.txt -p predictions.txt
```