import argparse
import json
import os
from tqdm import tqdm
from base_data_set import clean_nl
from my_ast import MyAst


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


def process(root_dir, mode, max_len, output_path):
    cnt = 0
    with open(root_dir + '{}.jsonl'.format(mode), 'r', encoding='utf-8') as f:
        asts = []
        nls = []
        idx = []
        for line in f.readlines():
            if cnt > 200:
                break
            ast_json = json.loads(line)

            nl = ' '.join(ast_json['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            idx.append(cnt)
            asts.append(ast_json['ast'])
            nls.append(nl)
            cnt = cnt + 1


    # is_skipped = PathExtract.collect_all_and_save(asts, output_path + 'paths.seq')
    #
    # asts = [ast for i, ast in enumerate(asts) if not is_skipped[i]]

    root_list = MyAst.process_ast(asts, max_size=max_len)

    MyAst.collect_matrices_and_save(root_list, output_path + 'un_split_matrices.npz', output_path + 'un_split_pot.jsonl', idx, nls)
    MyAst.collect_seq_and_save(root_list, output_path + 'un_split_sbt.jsonl', 'sbt', idx, nls)

    # root_list = MyAst.process_ast(asts, split_leaf=True, max_size=max_len)
    #
    # MyAst.collect_matrices_and_save(root_list, output_path + 'split_matrices.npz', output_path + 'split_pot.seq')
    # MyAst.collect_seq_and_save(root_list, output_path + 'split_sbt.seq', 'sbt')

    # skip code, nl with is_skipped
    # skip_code_and_nl_with_skip_id(data_dir, output_path, is_skipped)


if __name__ == '__main__':
    data_root_dir = '../dataset/'
    max_ast_len = 860

    data_modes = ['test', 'train', 'valid']

    for item in data_modes:
        processed_data_path = data_root_dir + item + '/'
        print('*' * 5, processed_data_path, '*' * 5)
        os.makedirs(processed_data_path, exist_ok=True)
        process(data_root_dir, item, max_ast_len, processed_data_path)


