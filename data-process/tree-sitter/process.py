import json
import GenerateAST
import sys
sys.setrecursionlimit(100000)

language = 'java'
print('start...')


def clean_ast(ast):
    return ast[62:-6]


def clean_split_ast(ast):
    pos = ast.find('(program(class_declaration(class)(identifier(A))(class_body({)')
    if pos != 0:
        pos = ast.find('(program(ERROR(class)(identifier(A))({)')
        if pos != 0:
            pos = ast.find('(ERROR(class)(identifier(A))({)')
            if pos != 0:
                pos = ast.find('(ERROR(ERROR(class)(identifier(A))({)')
                if pos != 0:
                    pos = ast.find('(ERROR(class_declaration(class)(identifier(A))(class_body({)')
                    if pos != 0:
                        print(ast)
                        return ast
                    else:
                        pos = 60
                else:
                    pos = 37
            else:
                pos = 31
        else:
            pos = 39
    else:
        pos = 62

    cnt_brk = 0
    for i in range(pos,len(ast)):
        if ast[i] == '(':
            cnt_brk = cnt_brk + 1
        if ast[i] == ')':
            cnt_brk = cnt_brk - 1
        if cnt_brk == 0:
            end = i
            break

    if cnt_brk == 0 :
        return ast[pos:end+1]
    else:     # cnt_brk > 0
        return ast[pos:]+')'*cnt_brk


idx = 0
# filtered = [9321,
# 9322,
# 9323,
# 9324,
# 9325,
# 9326,
# 9327,
# 9328,
# 9463,
# 9464,
# 9465,
# 9466,
# 9467,
# 9468,
# 9469,
# 9527,
# 9535,
# 9637,
# 47197,
# 131530]
# other
with open("D:\\ast_dataset\csn\original\\train.jsonl", encoding='UTF-8') as f:
    with open("D:\\ast_dataset\\csn\\binary_tree\\train_ast.jsonl", 'w', encoding='UTF-8') as f1:
        for line in f:
            idx = idx + 1
            # if idx in filtered:
            #     continue
            line = line.strip()
            js = json.loads(line)
            code = js['code']
            # code = js['func']
            ast = GenerateAST.get_ast(code, language)
            js['ast'] =clean_ast(ast)
            if(idx % 10000 == 0):
                print(js['ast'])
            f1.write(json.dumps(js) + '\n')
print(idx)
print('finish')




# split ast
# cnt = 0
# with open("D:\\ast_dataset\\bcb\\split_ast\\final_split_1.jsonl", encoding='UTF-8') as f:
#     with open("D:\\ast_dataset\\bcb\\split_ast\\data_ast.jsonl", 'w', encoding='UTF-8') as f1:
#         for line in f:
#             line = line.strip()
#             js = json.loads(line)
#             codes = js['func']
#             asts = []
#             for code in codes:
#                 ast = GenerateAST.get_ast(code, language)
#                 asts.append(clean_split_ast(ast))
#             js['asts'] = asts
#             f1.write(json.dumps(js) + '\n')
#             cnt = cnt + 1
# print(cnt)
# print('finish')