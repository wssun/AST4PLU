# Introduction
This subdirectory contrains code for Child-Sum TreeLSTM and N-ary TreeLSTM.

In our study, the input of TreeLSTM can be **AST**, **Binary Tree**, **Split AST**.

# Data
For how to process the BigCloneBench dataset into ASTs and preprocess the AST into Binary Tree and Split AST, please refer to the README in the data process directory.

The final dataset should be placed in the dataset directory. For the BigCloneBench data set, it should contain three files: train.jsonl, valid.jsonl, test.jsonl.


# Run
run the ``run.py`` file in the code directory.

Note that the code only supports single-gpu training. If you want to specify the GPU to use, modify ``os.environ['CUDA_VISIBLE_DEVICES'] = '0'`` in ``run.py/run_path.py``.


# Test
After running ``run.py``, you will get ``test_0.output`` and ``test_0.gold`` in the output directory.

You can use them to calculate METEOR and ROUGE-L.