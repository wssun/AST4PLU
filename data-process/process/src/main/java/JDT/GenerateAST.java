package JDT;

import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.FieldDeclaration;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.Modifier;
import org.eclipse.jdt.core.dom.NumberLiteral;
import org.eclipse.jdt.core.dom.PostfixExpression;
import org.eclipse.jdt.core.dom.PrefixExpression;
import org.eclipse.jdt.core.dom.PrimitiveType;
import org.eclipse.jdt.core.dom.QualifiedName;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.StringLiteral;
import org.eclipse.jdt.core.dom.Type;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.TextEdit;
import org.eclipse.jdt.core.dom.Block;
import org.eclipse.jdt.core.dom.BodyDeclaration;
import org.eclipse.jdt.core.dom.BooleanLiteral;
import org.eclipse.jdt.core.dom.CharacterLiteral;
import org.eclipse.jdt.core.dom.Statement;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import org.eclipse.jdt.core.dom.ExpressionStatement;
import org.eclipse.jdt.core.dom.Expression;
import org.eclipse.jdt.core.dom.Assignment;
import org.eclipse.jdt.core.dom.IfStatement;
import org.eclipse.jdt.core.dom.InfixExpression;
import org.eclipse.jdt.core.dom.Message;
import org.eclipse.jdt.core.dom.ReturnStatement;
import org.eclipse.jdt.core.dom.VariableDeclarationStatement;

public class GenerateAST {
	private static String ast;
	private static String masked_ast;
	
	public static class astVisitor extends ASTVisitor{
		public void preVisit(ASTNode node) {
			String type=node.nodeClassForType(node.getNodeType()).getSimpleName();
			if(node.nodeClassForType(node.getNodeType())==SimpleName.class)
			{
				SimpleName simpleName=(SimpleName)node;
				String value=simpleName.getIdentifier();
				ast = ast+ "("+type+"("+value+")";
			}else if(node.nodeClassForType(node.getNodeType())==Modifier.class)
			{
				Modifier modifier=(Modifier)node;
				String value=modifier.getKeyword().toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==NumberLiteral.class)
			{
				NumberLiteral numberLiteral=(NumberLiteral)node;
				String value=numberLiteral.getToken();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==StringLiteral.class)
			{
				StringLiteral stringLiteral=(StringLiteral)node;
				String value=stringLiteral.getLiteralValue();
				value=value.replaceAll(" ","");
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==CharacterLiteral.class)
			{
				CharacterLiteral characterLiteral=(CharacterLiteral)node;
				char value=characterLiteral.charValue();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==BooleanLiteral.class)
			{
				BooleanLiteral booleanLiteral=(BooleanLiteral)node;
				String value=booleanLiteral.toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==InfixExpression.class)
			{
				InfixExpression infixExpression=(InfixExpression)node;
				String value=infixExpression.getOperator().toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==PrefixExpression.class)
			{
				PrefixExpression prefixExpression=(PrefixExpression)node;
				String value=prefixExpression.getOperator().toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==PostfixExpression.class)
			{
				PostfixExpression postfixExpression=(PostfixExpression)node;
				String value=postfixExpression.getOperator().toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else if(node.nodeClassForType(node.getNodeType())==PrimitiveType.class)
			{
				PrimitiveType primitiveType=(PrimitiveType)node;
				String value=primitiveType.getPrimitiveTypeCode().toString();
				ast = ast+ "("+type+"("+value+")";
			}
			else ast = ast+ "("+type;
		}
		
		public void postVisit(ASTNode node) {
			ast = ast+ ")";
		}
	}
	
	public static String getAST(String code){
		ast="";
        ASTParser parser = ASTParser.newParser(AST.JLS8);
        Map options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_1_8, options);
        parser.setCompilerOptions(options);
        parser.setSource(code.toCharArray());
//    	parser.setKind(ASTParser.K_COMPILATION_UNIT);
//        parser.setKind(ASTParser.K_STATEMENTS);                     //split ast
//      parser.setKind(ASTParser.K_EXPRESSION);
        parser.setKind(ASTParser.K_CLASS_BODY_DECLARATIONS);
      
        //debug
//      ASTNode result = parser.createAST(new NullProgressMonitor());
//      if(result.nodeClassForType(result.getNodeType())==CompilationUnit.class)
//      {
//    	  CompilationUnit tmp = (CompilationUnit)result;
//    	  Message[] m = tmp.getMessages();
//    	  for(int i=0;i<m.length;++i)
//    	  {
//    		  System.out.println(m[i].getMessage());
//    	  }
//      }
        
//      CompilationUnit result = (CompilationUnit) parser.createAST(new NullProgressMonitor());
//      Block result = (Block) parser.createAST(new NullProgressMonitor());
//      CompilationUnit result = (CompilationUnit) parser.createAST(new NullProgressMonitor());
        TypeDeclaration result = (TypeDeclaration) parser.createAST(new NullProgressMonitor());
        result.accept(new astVisitor());
        return ast.substring(37);
//        return ast;
	}
	
	public static String getAST(String code,int kind) throws Exception{
		ast="";
        ASTParser parser = ASTParser.newParser(AST.JLS8);
        Map options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_1_8, options);
        parser.setCompilerOptions(options);
        parser.setSource(code.toCharArray());
        if(kind==0)
    	{
        	parser.setKind(ASTParser.K_COMPILATION_UNIT);
        	CompilationUnit result = (CompilationUnit) parser.createAST(new NullProgressMonitor());
        	result.accept(new astVisitor());
        	return ast.substring(37);
    	}
        else if(kind==1)
    	{
        	parser.setKind(ASTParser.K_STATEMENTS);       //split ast
        	Block result = (Block) parser.createAST(new NullProgressMonitor());
        	result.accept(new astVisitor());
        	return ast;
    	}
        else
        {
        	System.out.println("wrong kind code");
        	throw new Exception();
        }
	}
	
	public static class maskedVisitor extends ASTVisitor{
		public void preVisit(ASTNode node) {
			String type=node.nodeClassForType(node.getNodeType()).getSimpleName();
			if(node.nodeClassForType(node.getNodeType())==SimpleName.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}else if(node.nodeClassForType(node.getNodeType())==Modifier.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==NumberLiteral.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==StringLiteral.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==CharacterLiteral.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==BooleanLiteral.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==InfixExpression.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==PrefixExpression.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==PostfixExpression.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else if(node.nodeClassForType(node.getNodeType())==PrimitiveType.class)
			{
				ast = ast+ "("+type+"(<mask>)";
			}
			else ast = ast+ "("+type;
		}
		
		public void postVisit(ASTNode node) {
			ast = ast+ ")";
		}
	}
	
	public static String getMaskedAST(String code){
		ast="";
        ASTParser parser = ASTParser.newParser(AST.JLS8);
        Map options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_1_8, options);
        parser.setCompilerOptions(options);
        parser.setSource(code.toCharArray());
        parser.setKind(ASTParser.K_CLASS_BODY_DECLARATIONS);
        TypeDeclaration result = (TypeDeclaration) parser.createAST(new NullProgressMonitor());
        result.accept(new maskedVisitor());
        return ast.substring(37);
//        return ast;
	}
}
