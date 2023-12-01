import torch
from torch_geometric.data import Data
from tqdm import tqdm

from base_data_set import BaseASTDataSet


__all__ = ['FastASTDataSet']


class FastASTDataSet(BaseASTDataSet):
    def __init__(self, args, data_set_name):
        # print('Data Set Name : < Fast AST Data Set >')
        super(FastASTDataSet, self).__init__(args, data_set_name)
        self.max_par_rel_pos = args.max_par_rel_pos
        self.max_bro_rel_pos = args.max_bro_rel_pos
        self.data_type = args.data_type
        self.data_set_name = data_set_name
        self.final_dataset = self.convert_ast_to_edges()

    def convert_ast_to_edges(self):
        # print('building edges.')
        data = self.ast_data
        par_edge_data = self.matrices_data['parent']
        bro_edge_data = self.matrices_data['brother']
        data_type = self.data_type

        def edge2list(edges, edge_type):
            if edge_type == 'par':
                max_rel_pos = self.max_par_rel_pos
            if edge_type == 'bro':
                max_rel_pos = self.max_bro_rel_pos
            ast_len = min(len(edges), self.max_src_len)
            start_node = -1 * torch.ones((self.max_rel_pos + 1, self.max_src_len), dtype=torch.long)
            for key in edges.keys():
                if key[0] < self.max_src_len and key[1] < self.max_src_len:
                    value = edges.get(key)
                    if value > max_rel_pos and self.ignore_more_than_k:
                        continue
                    value = min(value, max_rel_pos)
                    start_node[value][key[1]] = key[0]

            start_node[0][:ast_len] = torch.arange(ast_len)
            return start_node

        final_dataset = []
        for i in tqdm(range(self.data_set_len)):
            example = data[i]
            nl = example.nl
            ast_seq = example.ast

            if data_type == 'sbt':
                ast_seq = ''.join(ast_seq)
            elif data_type == 'pot':
                ast_seq = ' '.join(ast_seq)
            else:
                print('Unknown data_type')
            par_edges = par_edge_data[i]
            bro_edges = bro_edge_data[i]

            par_edge_list = edge2list(par_edges, 'par')
            bro_edge_list = edge2list(bro_edges, 'bro')

            ast_ids = self.convert_ast_to_tensor(ast_seq, self.tokenizer)

            target_ids, target_mask = [], []
            if self.data_set_name == 'train':
                target_ids, target_mask = self.convert_nl_to_tensor(nl, self.tokenizer)

            item = Data(src_seq=ast_ids,
                        par_edges=par_edge_list,
                        bro_edges=bro_edge_list,
                        target_ids=target_ids,
                        target_mask=target_mask)

            final_dataset.append(item)

        print('{} items in final_dataset'.format(len(final_dataset)))

        return final_dataset

    def __getitem__(self, index):
        data = self.final_dataset[index]
        src_seq = data.src_seq
        par_edges = data.par_edges
        bro_edges = data.bro_edges
        target_ids = data.target_ids
        target_mask = data.target_mask

        return src_seq, par_edges, bro_edges, target_ids, target_mask
