package antlr;

import java.io.IOException;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.RuleContext;
import org.antlr.v4.runtime.TokenSource;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

public class GenerateAST {
	private static String ast;
    
	//Java
    public static String getJavaAST(String code)
    {
    	ast="";
    	ANTLRInputStream input = new ANTLRInputStream(code);
	    JavaLexer lexer = new JavaLexer(input);
	    CommonTokenStream tokens = new CommonTokenStream(lexer);
	    JavaParser parser = new JavaParser(tokens);
//	    ParserRuleContext ctx = parser.compilationUnit();
	    ParserRuleContext ctx = parser.classBodyDeclaration();
	    
	    traverseJavaAST(ctx);
	    return ast;
    }
    
    private static void traverseJavaAST(RuleContext ctx) {
        String ruleName = JavaParser.ruleNames[ctx.getRuleIndex()];
        ast = ast + "(" + ruleName;
        
        for (int i = 0; i < ctx.getChildCount(); i++) {
            ParseTree element = ctx.getChild(i);
            if (element instanceof RuleContext) {
            	traverseJavaAST((RuleContext) element);
            }
            else if(element instanceof TerminalNode)
            {
            	TerminalNode tn = (TerminalNode)element;
            	ast = ast + "(" + tn.getText()+")";
            }
            
        }
        ast = ast + ")";
    }
    
//    //Python
//    public static String getPythonAST(String code)
//    {
//    	ast="";
//    	ANTLRInputStream input = new ANTLRInputStream(code);
//    	PythonLexer lexer = new PythonLexer(input);
//	    CommonTokenStream tokens = new CommonTokenStream(lexer);
//	    PythonParser parser = new PythonParser(tokens);
//	    ParserRuleContext ctx = parser.single_input();
//	
//	    traversePythonAST(ctx);
////	    int len=ast.length();
////	    return ast.substring(0,len-23)+')';
//	    return ast;
//    }
//    
//    private static void traversePythonAST(RuleContext ctx) {
//        String ruleName = PythonParser.ruleNames[ctx.getRuleIndex()];
//        ast = ast + "(" + ruleName;
////        System.out.println(ast);
//        
//        for (int i = 0; i < ctx.getChildCount(); i++) {
//            ParseTree element = ctx.getChild(i);
//            if (element instanceof RuleContext) {
//            	traversePythonAST((RuleContext) element);
////                System.out.println(ast);
//            }
//            else if(element instanceof TerminalNode)
//            {
//            	TerminalNode tn = (TerminalNode)element;
//            	if(tn.getText()!="")ast = ast + "(" + tn.getText()+")"; 
////                System.out.println(ast);
//            }
//            
//        }
//        ast = ast + ")";
////        System.out.println(ast);
//    }
//    
//    //C++
//    public static String getCppAST(String code)
//    {
//    	ast="";
//    	ANTLRInputStream input = new ANTLRInputStream(code);
//    	CPP14Lexer lexer = new CPP14Lexer(input);
//	    CommonTokenStream tokens = new CommonTokenStream(lexer);
//	    CPP14Parser parser = new CPP14Parser(tokens);
//	    ParserRuleContext ctx = parser.translationUnit();
//	
//	    traverseCppAST(ctx);
//	    return ast;
//    }
//    
//    private static void traverseCppAST(RuleContext ctx) {
//        String ruleName = CPP14Parser.ruleNames[ctx.getRuleIndex()];
//        ast = ast + "(" + ruleName;
////        System.out.println(ast);
//        
//        for (int i = 0; i < ctx.getChildCount(); i++) {
//            ParseTree element = ctx.getChild(i);
//            if (element instanceof RuleContext) {
//            	traverseCppAST((RuleContext) element);
////                System.out.println(ast);
//            }
//            else if(element instanceof TerminalNode)
//            {
//            	TerminalNode tn = (TerminalNode)element;
//            	if(tn.getText()!="")ast = ast + "(" + tn.getText()+")"; 
////                System.out.println(ast);
//            }
//            
//        }
//        ast = ast + ")";
////        System.out.println(ast);
//    }
//    
//    //C
//    public static String getCAST(String code)
//    {
//    	ast="";
//    	ANTLRInputStream input = new ANTLRInputStream(code);
//    	CLexer lexer = new CLexer(input);
//	    CommonTokenStream tokens = new CommonTokenStream(lexer);
//	    CParser parser = new CParser(tokens);
//	    ParserRuleContext ctx = parser.translationUnit();
//	
//	    traverseCAST(ctx);
//	    return ast;
//    }
//    
//    private static void traverseCAST(RuleContext ctx) {
//        String ruleName = CParser.ruleNames[ctx.getRuleIndex()];
//        ast = ast + "(" + ruleName;
////        System.out.println(ast);
//        
//        for (int i = 0; i < ctx.getChildCount(); i++) {
//            ParseTree element = ctx.getChild(i);
//            if (element instanceof RuleContext) {
//            	traverseCAST((RuleContext) element);
////                System.out.println(ast);
//            }
//            else if(element instanceof TerminalNode)
//            {
//            	TerminalNode tn = (TerminalNode)element;
//            	if(tn.getText()!="")ast = ast + "(" + tn.getText()+")"; 
////                System.out.println(ast);
//            }
//            
//        }
//        ast = ast + ")";
////        System.out.println(ast);
//    }
}
