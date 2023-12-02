import json
import os
import tokenize
from io import StringIO

from tqdm import tqdm
import re


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)



def data2java_bcb(dataset_file, java_dir):  # txt
    file = open(dataset_file, encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        js = json.loads(line)
        idx = js['idx']
        code = js['func']

        # 去除注释
        # code = code.split('\n')
        # code = [c if c.strip().find('//')!=0 else '' for c in code]
        # code = '\n'.join(code)
        # 去除注释
        code = remove_comments_and_docstrings(code, 'java')

        new_file_path = java_dir + str(idx) + '.java'
        code = 'public class A' + str(idx) + '{\r\n' + code + '\r\n}'
        new_file = open(new_file_path, 'w', encoding='utf-8')
        new_file.write(code)
        new_file.close()
    file.close()


def data2java_csn(dataset_file, java_dir):  # txt
    file = open(dataset_file, encoding='utf-8')
    i = 0
    for line in file.readlines():
        line = line.strip()
        js = json.loads(line)
        code = js['code']
        i += 1

        # 去除注释
        code = remove_comments_and_docstrings(code, 'java')

        new_file_path = java_dir + str(i) + '.java'
        code = 'public class A' + str(i) + ' { \r\n' + code + '\r\n' + '}'
        new_file = open(new_file_path, 'w', encoding='utf-8')
        new_file.write(code)
        new_file.close()
    file.close()


def skip_empty_node(nodes):
    for _ in range((len(nodes)) // 4):
        for i in range(len(nodes)):
            next_nodes = nodes[i][6:]
            nodes[i] = nodes[i][:6]
            if len(next_nodes) and next_nodes[0] != '':
                # Handling the colon case
                for node in next_nodes:
                    if node.find(':') > 0:
                        temp = node
                        next_nodes.remove(node)
                        next_nodes += temp.split(':')
                # Skip the next empty node
                for node_id in next_nodes:
                    # The next node is an empty node
                    # 有的时候node_id类似于‘:31’，需要去掉‘:’
                    if node_id.find(':') == 0:
                        node_id = node_id[1:]
                    if node_id == '':
                        continue
                    if nodes[int(node_id) - 1][1] == '' and int(node_id) != len(nodes):
                        next_next_nodes = nodes[int(node_id) - 1][6:]
                        nodes[i] += next_next_nodes
                    else:
                        nodes[i] += [node_id]


def process_cfg_bcb(cfg_dir, java_dir, final_cfg_dir):   # read source code by line
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    for item in tqdm(cfg_file_list):
        if item.find('.txt') > 0:
            cfg_file = open(item, encoding='utf-8')
            cfg_id = item.split('.')[0].replace('A', '')
            try:
                java_file = open(java_dir + cfg_id + '.java', encoding='utf-8')
            except FileNotFoundError:
                print(cfg_id)
                continue
            source_code = java_file.readlines()
            nodes = cfg_file.read().split(';')
            for i in range(len(nodes)):
                nodes[i] = nodes[i].split(',')
            skip_empty_node(nodes)
            new_nodes = ''
            for i in range(len(nodes)):
                if i != 0 and i != len(nodes) - 1 and nodes[i][2] == '':
                    continue
                node_attrs = nodes[i]
                code_list = '' if node_attrs[1] == '' else source_code[int(node_attrs[1]) - 1:int(node_attrs[3])]
                code = ""
                for j in range(len(code_list)):
                    code_list[j] = code_list[j].replace('\t', '').replace('\n', '').strip()
                    code += code_list[j]
                    if j != len(code_list) - 1:
                        code += "\n"
                next_nodes = list(set(node_attrs[6:]))
                # Generate a new CFG graph node
                if i != len(nodes):
                    new_nodes += json.dumps(
                        {"id": str(i + 1), "source_code": code.replace('\n', ''), "next_nodes": next_nodes}) + '\n'
            new_cfg_file = open(final_cfg_dir + cfg_id + '.json', 'w',
                                encoding='utf-8')
            new_cfg_file.write(new_nodes)
            new_cfg_file.close()
            java_file.close()
            cfg_file.close()


def process_cfg_csn(cfg_dir, java_dir, final_cfg_dir):   # read source code by line
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    processed = []
    for item in tqdm(cfg_file_list):
        if item.find('.txt') > 0:
            cfg_file = open(item, encoding='utf-8')
            cfg_id = item.split('.')[0].replace('A', '')
            if cfg_id in processed:   # 某些文件（如valid-A4168包含多个函数，只取第一个方法）
                continue
            else:
                processed.append(cfg_id)
            try:
                java_file = open(java_dir + cfg_id + '.java', encoding='utf-8')
            except FileNotFoundError:
                print(cfg_id)
                continue
            source_code = java_file.readlines()
            nodes = cfg_file.read().split(';')
            for i in range(len(nodes)):
                nodes[i] = nodes[i].split(',')
            skip_empty_node(nodes)
            new_nodes = ''
            for i in range(len(nodes)):
                if i != 0 and i != len(nodes) - 1 and nodes[i][2] == '':
                    continue
                node_attrs = nodes[i]
                code_list = '' if node_attrs[1] == '' else source_code[int(node_attrs[1]) - 1:int(node_attrs[3])]
                code = ""
                for j in range(len(code_list)):
                    code_list[j] = code_list[j].replace('\t', '').replace('\n', '').strip()
                    code += code_list[j]
                    if j != len(code_list) - 1:
                        code += "\n"
                next_nodes = list(set(node_attrs[6:]))
                # Generate a new CFG graph node
                if i != len(nodes):
                    new_nodes += json.dumps(
                        {"id": str(i + 1), "source_code": code.replace('\n', ''), "next_nodes": next_nodes}) + '\n'
            new_cfg_file = open(final_cfg_dir + cfg_id + '.json', 'w',
                                encoding='utf-8')
            new_cfg_file.write(new_nodes)
            new_cfg_file.close()
            java_file.close()
            cfg_file.close()


def add_head_bcb(code_split_dir, source_path, new_split_path):  # The method header is also added to the code snippet
    code_split_list = os.listdir(code_split_dir)
    source_code = {}
    with open(source_path, encoding='UTF-8') as f1:
        for line in f1:
            line = line.strip()
            js = json.loads(line)
            source_code[js['idx']] = js['func']
    new_file = open(new_split_path, 'w', encoding='utf-8')
    os.chdir(code_split_dir)
    for f in tqdm(code_split_list):
        file = open(f, encoding='utf-8')
        idx = f.replace('.txt', '')
        lines = file.readlines()
        if len(lines) == 0:
            line = ''
        else:
            line = lines[0]
        s_line = source_code[idx].strip()

        # 有些代码前几行是“@xxxx”, 不是方法名
        line_code = s_line.split('\n')
        i = 0
        while(True):
            head = line_code[i].strip()
            if(head.find('@')==0):
                i = i + 1
            else:
                break
        end = head.find('(')
        cnt = 0
        for j in range(end, len(head)):
            if head[j] == '(':
                cnt = cnt + 1
            elif head[j] == ')':
                cnt = cnt - 1
            if cnt == 0:
                end = j
                break
        s_line = head[0:end + 1] + ';'

        # 提取方法名
        # s_line = s_line[0:s_line.find(')')+1] + ';'

        line = s_line + line
        line = line.split('<sep>')
        item = {}
        item['idx'] = idx
        item['func'] = line
        new_file.write(json.dumps(item) + '\n')
    new_file.close()


def add_head_csn(code_split_dir, source_path, new_split_path):  # The method header is also added to the code snippet
    new_file = open(new_split_path, 'w', encoding='utf-8')

    idx = 0
    with open(source_path, encoding='UTF-8') as f1:
        for line in f1:
            line = line.strip()
            js = json.loads(line)
            source_code = js['code']
            idx = idx + 1

            # 某些file understand解析失败，无法生成cfg
            try:
                file = open(code_split_dir+str(idx)+'.txt', encoding='utf-8')
            except:
                continue

            lines = file.readlines()
            if len(lines) == 0:
                line = ''
            else:
                line = lines[0]
            s_line = source_code.strip()

            # 有些代码前几行是“@xxxx”, 不是方法名
            line_code = s_line.split('\n')
            i = 0
            while(True):
                head = line_code[i].strip()
                if(head.find('@')==0):
                    i = i + 1
                else:
                    break
            line_code = line_code[i:]
            new_code = ' '.join(line_code)    # 去掉了‘@xxx’的源代码
            end = new_code.find('(')
            cnt = 0
            for j in range(end, len(new_code)):
                if new_code[j] == '(':
                    cnt = cnt + 1
                elif new_code[j] == ')':
                    cnt = cnt - 1
                if cnt == 0:
                    end = j
                    break
            s_line = new_code[0:end + 1] + ';'

            line = s_line + line
            line = line.split('<sep>')
            js['func'] = line
            new_file.write(json.dumps(js) + '\n')

    new_file.close()



def clean_code(code):
    code = code.replace('\n', '')
    code = code.replace('default:', '')
    if code.find('throw') >= 0:
        return ''
    while code.find('case') >= 0:
        # code = code[code.rfind(':') + 1:]
        code = code[code.find(':') + 1:]
        if code.find(':') < 0:
            break
    if code.find('switch (') == 0 and code.find('{'):
        code = code[code.find('{')+1:code.find('}')]
    if code.find('for') >= 0 or code.find('while') >= 0:
        code += '{}'

    return code


def clean_head(head):
    first = head[0:head.find('(')]
    second = head[head.find('('):]
    words = first.split(' ')
    name = words[-1]
    head = 'void ' + name + second
    return head


def process_split_code(code_split_path, final_code_split_file):
    final_code_split = open(final_code_split_file, 'w', encoding='utf-8')
    with open(code_split_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code_list = js['func']
            new_code = []
            head = clean_head(code_list[0])     # 注释掉就是final_split_2或final_split_3
            new_code.append(head)

            if len(code_list) > 1:
                code_list = code_list[1:]
                # Add method headers to each code segment to generate AST
                for code in code_list:
                    if code.replace('\n', '').strip() != '':
                        code = clean_code(code)    # 注释掉就是final_split_2
                        if len(code) <= 0: continue
                        code = head.replace(';', '') + '{' + code + '}'
                        new_code.append(code)

            js['func']=new_code
            final_code_split.write(json.dumps(js) + '\n')
    final_code_split.close()
    



if __name__ == "__main__":
    csn_file = 'test'

    # 1. Run data2java function
    # data2java_bcb('D:\\ast_dataset\\bcb\\func\\data.jsonl','D:\\ast_dataset\\bcb\\split_ast\\files\\')
    # data2java_csn('D:\\ast_dataset\\csn\\original\\{}.jsonl'.format(csn_file), 'D:\\ast_dataset\\csn\\split_ast\\files\\{}\\'.format(csn_file))

    # 2. open 'D:\\ast_dataset\\bcb\\split_ast\\files\\' in scitools understand to get ‘D:\ast_dataset\bcb\split_ast\files.udb’
    #   如果已有‘D:\ast_dataset\bcb\split_ast\files.udb’文件可以直接打开，会自己更新

    # 3. run in cmd ‘D:\SciTools\bin\pc - win64 > uperl.exe D:\ast_dataset\bcb\split_ast\test.pl -db D:\ast_dataset\bcb\split_ast\files.udb’
    #   生成结果在'D:\\ast_dataset\\bcb\\split_ast\\cfgs' 有一些多余的文件需要手动删掉（这一步还是9124个文件）
    #   run in cmd ‘D:\SciTools\bin\pc - win64 > uperl.exe D:\ast_dataset\csn\split_ast\files\valid.pl -db D:\ast_dataset\csn\split_ast\files\valid.udb
    #   有一些多余的文件需要手动删掉后还是有多出的文件，因为有些数据不止包含一个函数（不用删掉）

    # 4. Run process_cfg function
    # process_cfg_bcb('D:\\ast_dataset\\bcb\\split_ast\\cfgs', 'D:\\ast_dataset\\bcb\\split_ast\\files\\', 'D:\\ast_dataset\\bcb\\split_ast\\final_cfgs\\')
    # 'D:\\ast_dataset\\bcb\\split_ast\\final_cfgs\\' 中的‘img’文件夹需要手动删掉（这一步还是9124个文件）
    # process_cfg_csn('D:\\ast_dataset\\csn\\split_ast\\cfgs\\{}\\'.format(csn_file),
    #             'D:\\ast_dataset\\csn\\split_ast\\files\\{}\\'.format(csn_file),
    #             'D:\\ast_dataset\\csn\\split_ast\\final_cfgs\\{}\\'.format(csn_file))
    # process_cfg_csn('D:\\ast_dataset\\csn\\split_ast\\error_cfgs\\{}\\'.format(csn_file),
    #             'D:\\ast_dataset\\csn\\split_ast\\files\\{}\\'.format(csn_file),
    #             'D:\\ast_dataset\\csn\\split_ast\\error_final_cfgs\\')

    # 5. run dominator_tree.py  （结果在'D:\\ast_dataset\\bcb\\split_ast\\code_split\\'）
    # 部分文件会无法生成, 因为CFG有多个入口，把final_cfgs文件夹里的对应json文件里多余的入口节点（整个文件中id只出现了一次的）删掉

    # 6. Run add_head function
    add_head_bcb('D:\\ast_dataset\\bcb\\split_ast\\code_split\\', 'D:\\ast_dataset\\bcb\\func\\data.jsonl', 'D:\\ast_dataset\\bcb\\split_ast\\final_split.jsonl')
    # add_head_csn('D:\\ast_dataset\\csn\\split_ast\\code_split\\{}\\'.format(csn_file), 'D:\\ast_dataset\\csn\\original\\{}.jsonl'.format(csn_file),
    #          'D:\\ast_dataset\\csn\\split_ast\\final_split_{}.jsonl'.format(csn_file))

    # 7. Run process_split_code function
    process_split_code('D:\\ast_dataset\\bcb\\split_ast\\final_split.jsonl','D:\\ast_dataset\\bcb\\split_ast\\final_split_1.jsonl')
    # process_split_code('D:\\ast_dataset\\csn\\split_ast\\final_split_{}.jsonl'.format(csn_file),
    #                    'D:\\ast_dataset\\csn\\split_ast\\final_split_{}_1.jsonl'.format(csn_file))

