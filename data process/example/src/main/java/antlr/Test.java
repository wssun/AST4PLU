package antlr;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;

public class Test {

    public static void main(String args[]) throws IOException {
	        String inputString = "class Test {public String extractFor(Integer id){LOG.debug(\"Extracting method with ID:{}\", id);return requests.remove(id);}}";
//    	String inputString="class Test {public int add(int a,int b){if(a==0)return a;else return a+b;}}";
    	ANTLRInputStream input = new ANTLRInputStream(inputString);
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        JavaParser parser = new JavaParser(tokens);
//	        ParserRuleContext ctx = parser.classDeclaration();
        ParserRuleContext ctx = parser.compilationUnit();

        printAST(ctx, false,0);
        printAST2(ctx, false);
//        String code="def make_cursor(path, init_statements=(), _connectioncache={}):\r\n    \"\"\"Returns a cursor to the database, making new connection if not cached.\"\"\"\r\n    connection = _connectioncache.get(path)\r\n    if not connection:\r\n        is_new = not os.path.exists(path) or not os.path.getsize(path)\r\n        try: is_new and os.makedirs(os.path.dirname(path))\r\n        except OSError: pass\r\n        connection = sqlite3.connect(path, isolation_level=None,\r\n            check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)\r\n        for x in init_statements or (): connection.execute(x)\r\n        try: is_new and \":memory:\" not in path.lower() and os.chmod(path, 0707)\r\n        except OSError: pass\r\n        connection.row_factory = lambda cur, row: dict(sqlite3.Row(cur, row))\r\n        _connectioncache[path] = connection\r\n    return connection.cursor()";
//        String ast=GenerateAST.getPythonAST(code);
//        System.out.println(ast);
    }

    private static void printAST(RuleContext ctx, boolean verbose, int indentation) {
        String ruleName = JavaParser.ruleNames[ctx.getRuleIndex()];
        for (int i = 0; i < indentation; i++) {
            System.out.print("  ");
        }
        System.out.println(ruleName + " -> " + ctx.getText());
        
        for (int i = 0; i < ctx.getChildCount(); i++) {
            ParseTree element = ctx.getChild(i);
            if (element instanceof RuleContext) {
                printAST((RuleContext) element, verbose, indentation + 1);
            }
            else if(element instanceof TerminalNode)
            {
            	TerminalNode tn = (TerminalNode)element;
            	for (int j = 0; j <= indentation; j++) {
                    System.out.print("  ");
                }
                System.out.println(tn.getText()); 
            }
        }
    }
    
    private static void printAST2(RuleContext ctx, boolean verbose) {
        String ruleName = JavaParser.ruleNames[ctx.getRuleIndex()];
        System.out.print("("+ruleName);
        
        for (int i = 0; i < ctx.getChildCount(); i++) {
            ParseTree element = ctx.getChild(i);
            if (element instanceof RuleContext) {
                printAST2((RuleContext) element, verbose);
            }
            else if(element instanceof TerminalNode)
            {
            	TerminalNode tn = (TerminalNode)element;
            	System.out.print("(" + tn.getText()+")"); 
            }
            
        }
        System.out.print(")");
    }

}
