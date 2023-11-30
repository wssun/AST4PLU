# -- coding:utf-8 --
from tree_sitter import Language, Parser

Language.build_library(
        # Store the library in the `build` directory
        'my-languages.so',

        # Include one or more languages
        ['tree-sitter-java', 'tree-sitter-cpp', 'tree-sitter-python']
    )

JAVA_LANGUAGE = Language('my-languages.so', 'java')
CPP_LANGUAGE = Language('my-languages.so', 'cpp')
PYTHON_LANGUAGE = Language('my-languages.so', 'python')

punctuation = ['(', ')', '{', '}', '[', ']', ';', ',']
type_literal = ['boolean_type', 'void_type']
string_literal = ['string_literal', 'string', 'char_literal']
python_literal = ['string', 'integer', 'float']

ast = ''

# 尾递归
import sys

class TailCallException(BaseException):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

def tail_call_optimized(func):
    def _wrapper(*args, **kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back and f.f_code == f.f_back.f_back.f_code:
            raise TailCallException(args, kwargs)

        else:
            while True:
                try:
                    return func(*args, **kwargs)
                except TailCallException as e:
                    args = e.args
                    kwargs = e.kwargs
    return _wrapper


def get_ast(code, language, output_punctuation=1):
    global ast
    ast = ''
    if (language == 'cpp'):
        lang = CPP_LANGUAGE
    elif (language == 'java'):
        code = 'class A{' + code + '}'
        lang = JAVA_LANGUAGE
    else:
        lang = PYTHON_LANGUAGE
    parser = Parser()
    parser.set_language(lang)
    tree = parser.parse(bytes(code, 'utf8'))
    cursor = tree.walk()

    make_move(cursor, "down", output_punctuation)
    return ast

@tail_call_optimized
def make_move(cursor, move, output_punctuation):
    global ast
    #递归遍历AST
    type = cursor.node.type
    if type == '(':
        type = '<left_bracket_5175241>'
    if type == ')':
        type = '<right_bracket_5175241>'

    is_type_literal = 0  # java中没有子节点的xx_type
    is_string_literal = 0  # (C++和python)不输出string_literal的子节点（两个双引号）
    is_python_literal = 0  # python中的xx_literal节点
    if (type in type_literal):
        is_type_literal = 1
    if (type in string_literal):
        is_string_literal = 1
    if (type in python_literal):
        is_python_literal = 1

    if ('identifier' in type or 'literal' in type or type == 'primitive_type' or is_python_literal == 1 or is_type_literal == 1):
        # type=type+'_'+(str)(cursor.node.text)[1:]
        type = type + '(' + (str)(cursor.node.text)[2:-1] + ')'
    # type=type+' '

    output = 1
    if (output_punctuation == 0 and type in punctuation):
        output = 0

    if (move == "down"):
        if (output == 1):
           ast = ast + '(' + type
        if (is_string_literal == 0 and cursor.goto_first_child()):
            make_move(cursor, "down", output_punctuation)
        elif (cursor.goto_next_sibling()):
            if (output == 1):
                ast = ast + ')'
            make_move(cursor, "right", output_punctuation)
        elif (cursor.goto_parent()):
            ast = ast + ')'
            make_move(cursor, "up", output_punctuation)
    elif (move == "right"):
        if (output == 1):
            ast = ast + '(' + type
        if (is_string_literal == 0 and cursor.goto_first_child()):
            make_move(cursor, "down", output_punctuation)
        elif (cursor.goto_next_sibling()):
            if (output == 1):
                ast = ast + ')'
            make_move(cursor, "right", output_punctuation)
        elif (cursor.goto_parent()):
            if (output == 1):
                ast = ast + ')'
            make_move(cursor, "up", output_punctuation)
    elif move == "up":
        ast = ast + ')'
        if (cursor.goto_next_sibling()):
            make_move(cursor, "right", output_punctuation)
        elif (cursor.goto_parent()):
            make_move(cursor, "up", output_punctuation)


if __name__ == '__main__':
    # src不能为空
    src1 = 'int sum(int a,int b)\
{\
    int res=0;\
    for(int i=a;i<b;++i)res=res+i;\
    return res;\
}'

    src2 = 'public int max(int a,int b)\
{\
	if(a>b)return a;\
	else return b;\
}'

    src3 = "def sum(a, b):\r\n" + \
			"    res = 0\r\n" + \
			"    for i in range(a,b):\r\n" + \
			"        res = res + i\r\n" + \
			"    return res"

    clean = "def factorial(num):\r\n" + \
            "    if num == 0: return 1\r\n" + \
            "    factorial = 1\r\n" + \
            "    for i in range(1, num + 1):\r\n" + \
            "        factorial = factorial * i\r\n" + \
            "    return factorial"

    baseline = "def factorial(num):\r\n" + \
               "    import logging\r\n" + \
               "    for i in range(0):\r\n" + \
               "        logging.info(\"Test message:aaaaa\")\r\n" + \
               "    if num == 0: return 1\r\n" + \
               "    factorial = 1\r\n" + \
               "    for i in range(1, num + 1):\r\n" + \
               "        factorial = factorial * i\r\n" + \
               "    return factorial\r\n"

    our = "def factorial(num):\r\n" + \
            "    if num == 0: return 1\r\n" + \
            "    factorial = 1\r\n" + \
            "    for i_rb in range(1, num + 1):\r\n" + \
            "        factorial = factorial * i_rb\r\n" + \
            "    return factorial"

    # new = baseline + '\n\n' + our
    # print(new)

    # print(get_ast(src1,'cpp'))
    # print(get_ast(src2,'java'))
    # print(get_ast(new,'python'))

    split = "    private String postXml(String url, String soapAction, String xml) {\r\n"\
				"        try {\r\n"\
				"            URLConnection conn = new URL(url).openConnection();\r\n"\
				"            if (conn instanceof HttpURLConnection) {\r\n"\
				"                HttpURLConnection hConn = (HttpURLConnection) conn;\r\n"\
				"                hConn.setRequestMethod(\"POST\");\r\n"\
				"            }\r\n"\
				"        } catch (IOException e) {\r\n"\
				"            throw new RuntimeException(e);\r\n"\
				"        }\r\n"\
				"    }\r\n"
    print(get_ast(split, 'java'))