# Introduction
This subdirectory contrains code for Child-Sum TreeLSTM and N-ary TreeLSTM.

In our study, the input of sequence models can be **AST**, **Binary Tree**, **Split AST**.

# Data
For how to process the BigCloneBench dataset into ASTs and preprocess the AST into Binary Tree and Split AST, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: data.jsonl, train.txt, valid.txt, test.txt.


# Run
**If the input is Split AST**, run the ``run_split.py`` file in the code directory, 
**else** run the ``run.py`` file in the code directory.

Note that the code only supports single-gpu training. If you want to specify the GPU to use, modify ``os.environ['CUDA_VISIBLE_DEVICES'] = '0'`` in ``run.py/run_path.py``.


# Test
After running ``run.py/run_split.py``, you will get ``predictions.txt`` in the output directory.

You can move ``predictions.txt`` to the evaluator directory and run the following command to get the evaluation result.

```buildoutcfg
cd ./evaluator
python evaluator -a ../dataset/test.txt -p predictions.txt
```