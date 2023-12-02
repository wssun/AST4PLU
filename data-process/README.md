# Introduction
This directory contains files to generate AST and preprocess AST.

**AST Parsers** - We use JDT, SrcML, Antlr and Tree-sitter in our experiment. You can find a simple guide about how to use these parsers below.

**AST Preprocessing** - We provide the code to process AST into BFS, SBT, SBT w/o Token, AST Path, Binary Tree and Split AST in the ways decscribed in our paper.

The table below shows the combination of AST parsers and preprocessing methods used in our experiment.

| |AST|BFS|SBT|SBT w/o Token|AST Path|Binary Tree|Split AST|
|---|---|---|---|---|---|---|---|
|JDT|√|√|√|√|√|-|-|
|SrcML|√|-|√|-|-|-|-|
|Antlr|√|-|√|-|-|-|-|
|Tree-sitter|√|-|√|-|-|√|√|

The table below shows the explanation of some unified parameters.

|Parameter|Explanation|
|---|---|
|``FILE_PATH`` or ``file_path``|path to original data file (a json file) in BigCloneBench or CodeSearchNet|
|``JSON_FILE_PATH``|path to the result data file (a json file)|
|``TEMP_CODE_PATH``|path to a temporary file without suffix, such as "D:\\ast_dataset\\csn\\func_sbt_ast\\srcml\\temp"|
|``TEMP_AST_PATH``|path to a xml file, such as "D:\\ast_dataset\\csn\\func_sbt_ast\\srcml\\ast.xml"|
|``AST_FILE_PATH`` or ``ast_file_path``|path to a intermediate data file (a json file) generated using Tree-sitter|
|``MAX_SIZE``|the max number of data in the dataset|

# Install AST parsers
The steps for installing JDT, SrcML, Antlr and Tree-sitter can be easily found online.

Note that:
- JDT can obly be used in Eclipse
- Antlr should be installed in Java environment
- Tree-sitter should be installed in Python environment

We recorded our installation steps, problems we encountered and how we fixed them in ``JDT.pdf``, ``SrcML.pdf``, ``Antlr.pdf`` and ``Tree-sitter.pdf`` as a reference to you. (These pdfs are written in Chinese.)

The code that use JDT, SrcML and Antlr to process datasets are implemented in Java.
However part of the code that use Tree-sitter to process datasets are implemented in Python.
We introduce the steps to process datasets using the four AST parsers below.

# Process dataset using JDT, SrcML and Antlr
The code is in ``./process/src/main/java/process/`` directory.
Specific file names are shown in the Table below where {DATASET} can be ``bcb``(BigCloneBench) or ``csn``(CodeSearchNet).

| |AST|BFS|SBT|SBT w/o Token|AST Path|Binary Tree|Split AST|
|---|---|---|---|---|---|---|---|
|JDT|JDT_for_{DATASET}.java|JDT_for_{DATASET}.java|JDT_for_{DATASET}.java|JDT_for_{DATASET}.java|ASTPath_for_{DATASET}.java|-|-|
|SrcML|SrcML_for_{DATASET}.java|-|SrcML_for_{DATASET}.java|-|-|-|-|
|Antlr|Antlr_for_{DATASET}.java|-|Antlr_for_{DATASET}.java|-|-|-|-|

## Original AST, BFS, SBT
Run the corresponding file.

## SBT w/o Token
Similar to SBT, the code for processing BigCloneBench and CodeSearchNet into SBT w/o Token is ``JDT_for_bcb.java`` and ``JDT_for_csn.java`` respectively.
However you have to change the following code
```
                try {
                	ast_seq=GenerateAST.getAST(code);
//                	ast_seq=GenerateAST.getMaskedAST(code);
                }
```

into
```
                try {
//                 	ast_seq=GenerateAST.getAST(code);
               	    ast_seq=GenerateAST.getMaskedAST(code);
                }
```

## AST Path
If you want to control the max width and max length of AST Path, you can change the paratemers of
```TreeTools.getASTPath(Tree tree, int maxLen, int maxWid)```.
For example, we use ```TreeTools.getASTPath(ast, 8, 2)```, which sets maxLen=8 and maxWid=2.


# Process dataset using Tree-sitter
First, we use Tree-sitter to generate an intermediate output, which is a sequence similar to SBT, to represent AST (This part of code is written in Python).
Then we use the same code as JDT, SrcML and Antlr for further processing (This part of code is wwritten in java).

The former part of code is in ``./tree-sitter`` directory.
The latter part of code is in ``./process/src/main/java/process/`` directory.
Specific file names are shown in the Table below where {DATASET} can be ``bcb``(BigCloneBench) or ``csn``(CodeSearchNet).

| |AST|BFS|SBT|SBT w/o Token|AST Path|Binary Tree|Split AST|
|---|---|---|---|---|---|---|---|
|Tree-sitter(Python)|process.py|-|process.py|-|-|-|split_ast.py/dominator_tree.py/process.py|
|Tree-sitter(Java)|Treesitter_for_{DATASET}.java|-|Treesitter_for_{DATASET}.java|-|-|-|SplitAST_for_{DATASET}.java|

## Original AST, SBT
Run the corresponding file.

## Binary Tree
Similar to AST, the code for processing BigCloneBench and CodeSearchNet into Binary Tree is ``Treesitter_for_bcb.java`` and ``Treesitter_for_csn.java`` respectively.
However, in ``Treesitter_for_bcb.java``, you have to change the following code
```
                Tree ast=TreeTools.stringToTree(ast_seq);
                TreeToJSON.toJSON(ast,0);
                JSONArray tree=TreeToJSON.getJSONArray();
                String sbt=TreeTools.treeToSBT(ast);
//                 List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
//         		   BinaryTree bn=TreeTools.TreeToBinary(ast);
//                 BinaryToJSON.toJSON(bn,0);
//                 JSONArray tree=BinaryToJSON.getJSONArray();
```
into
```
                Tree ast=TreeTools.stringToTree(ast_seq);
//                 TreeToJSON.toJSON(ast,0);
//                 JSONArray tree=TreeToJSON.getJSONArray();
//                 String sbt=TreeTools.treeToSBT(ast);
//                 List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
        		BinaryTree bn=TreeTools.TreeToBinary(ast);
                BinaryToJSON.toJSON(bn,0);
                JSONArray tree=BinaryToJSON.getJSONArray();
```
and comment out the sentence ``tr.put("sbt",sbt);`

Similarly, in ``Treesitter_for_csn.java``, you have to change the following code
```
                Tree ast=TreeTools.stringToTree(ast_seq);
                TreeToJSON.toJSON(ast,0);
                JSONArray tree=TreeToJSON.getJSONArray();
                List<String> sbt=TreeTools.treeToSBTArrayBrackets(ast);
//              List<String> nonleaf=TreeTools.treeToNonLeaf(ast);
//        		BinaryTree bn=TreeTools.TreeToBinary(ast);
//              BinaryToJSON.toJSON(bn,0);
//              JSONArray tree=BinaryToJSON.getJSONArray();
```
into
```
                Tree ast=TreeTools.stringToTree(ast_seq);
//              TreeToJSON.toJSON(ast,0);
//              JSONArray tree=TreeToJSON.getJSONArray();
//              List<String> sbt=TreeTools.treeToSBTArrayBrackets(ast);
//              List<String> nonleaf=TreeTools.treeToNonLeaf(ast);
       		    BinaryTree bn=TreeTools.TreeToBinary(ast);
                BinaryToJSON.toJSON(bn,0);
                JSONArray tree=BinaryToJSON.getJSONArray();
```
and comment out the sentence ``tr.put("sbt",sbt);`

## Split AST

### Get Split Code
You can follow the steps in the comment in ``split_ast.py`` to get Split Code.

### Get Split AST
TODO



# RQ1: Differences among ASTs
The code is in ``./process/src/main/java/statistic`` directory. Specific file names are shown in the Table below.

|AST Parser|File Name|
|---|---|---|---|---|---|---|---|
|JDT|JDT.java|
|SrcML|SrcML.java|
|Antlr|Antlr.java|
|Tree-sitter|TreeSitter.java|

Run the corresponding file and you will get Tree Size, Tree Depth, Branch Factor, Unique Types and Unique Tokens statistics of the dataset.


