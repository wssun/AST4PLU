import argparse
import json
import os
from tqdm import tqdm
from base_data_set import clean_nl
from my_ast import MyAst, PathExtract

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='../dataset/', type=str, help="path to the dataset directory")
parser.add_argument('-max_ast_len', default=860, type=int, help="max number of AST nodes")


def skip_code_and_nl_with_skip_id(data_dir, output_dir, is_skipped):
    # skip data.
    nls = []
    with open(data_dir + 'nl.original', 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            if not is_skipped[line_index]:
                nls.append(line)

    codes = []
    with open(data_dir + 'code.seq', 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            if not is_skipped[line_index]:
                codes.append(line)

    # write to output_dir
    data_size = len(nls)

    with open(output_dir + 'nl.original', 'w') as f:
        for index, nl in tqdm(enumerate(nls), desc='skip nl'):
            nl = clean_nl(nl)
            nl = ' '.join(nl)
            if index < data_size-1:
                nl = nl + '\n'
            f.write(nl)

    with open(output_dir + 'code.seq', 'w') as f:
        for index, code in tqdm(enumerate(codes), desc='skip code'):
            f.write(code)


def process(data_dir, max_len, output_path):
    cnt = 0
    with open(data_dir + 'data.jsonl', 'r', encoding='utf-8') as f:
        asts = []
        idx = []
        for line in f.readlines():
            cnt = cnt + 1
            # if cnt < 150:
            #     continue
            ast_json = json.loads(line)
            idx.append(ast_json['idx'])
            asts.append(ast_json['ast'])
            # if cnt > 3000:
            #     break

    # is_skipped = PathExtract.collect_all_and_save(asts, output_path + 'paths.seq')
    #
    # asts = [ast for i, ast in enumerate(asts) if not is_skipped[i]]

    root_list = MyAst.process_ast(asts, max_size=max_len)

    MyAst.collect_matrices_and_save(root_list, output_path + 'un_split_matrices.npz', output_path + 'un_split_pot.jsonl', idx)
    MyAst.collect_seq_and_save(root_list, output_path + 'un_split_sbt.jsonl', 'sbt', idx)

    # root_list = MyAst.process_ast(asts, split_leaf=True, max_size=max_len)
    #
    # MyAst.collect_matrices_and_save(root_list, output_path + 'split_matrices.npz', output_path + 'split_pot.seq')
    # MyAst.collect_seq_and_save(root_list, output_path + 'split_sbt.seq', 'sbt')

    # skip code, nl with is_skipped
    # skip_code_and_nl_with_skip_id(data_dir, output_path, is_skipped)


if __name__ == '__main__':
    args = parser.parse_args()
    data_set_dir = args.data_dir
    max_ast_len = args.max_ast_len
    print('*' * 5, 'Process ', data_set_dir, '*' * 5)
    process(data_set_dir, max_ast_len, data_set_dir)

    # data_sets = ['test/', 'valid/', 'train/']

    # if args.process:
    #     for data_set in data_sets:
    #         data_path = data_set_dir + data_set
    #         print('*' * 5, 'Process ', data_path, '*' * 5)
    #         processed_path = data_set_dir + 'processed/' + data_set
    #         os.makedirs(processed_path, exist_ok=True)
    #         process(data_path, max_ast_len, processed_path)

    # if args.make_vocab:
    #     create_vocab(data_dir=data_set_dir + 'processed/')

