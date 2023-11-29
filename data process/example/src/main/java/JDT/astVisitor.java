package JDT;

import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.BooleanLiteral;
import org.eclipse.jdt.core.dom.InfixExpression;
import org.eclipse.jdt.core.dom.Modifier;
import org.eclipse.jdt.core.dom.NumberLiteral;
import org.eclipse.jdt.core.dom.PostfixExpression;
import org.eclipse.jdt.core.dom.PrefixExpression;
import org.eclipse.jdt.core.dom.PrimitiveType;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.StringLiteral;

public class astVisitor extends ASTVisitor{
	public void preVisit(ASTNode node) {
		String type=node.nodeClassForType(node.getNodeType()).getSimpleName();
		if(node.nodeClassForType(node.getNodeType())==SimpleName.class)
		{
			SimpleName simpleName=(SimpleName)node;
			String value=simpleName.getIdentifier();
			System.out.print("("+type+"("+value+")");
		}else if(node.nodeClassForType(node.getNodeType())==Modifier.class)
		{
			Modifier modifier=(Modifier)node;
			String value=modifier.getKeyword().toString();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==NumberLiteral.class)
		{
			NumberLiteral numberLiteral=(NumberLiteral)node;
			String value=numberLiteral.getToken();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==StringLiteral.class)
		{
			StringLiteral stringLiteral=(StringLiteral)node;
			String value=stringLiteral.getLiteralValue();
			value=value.replaceAll(" ","");
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==BooleanLiteral.class)
		{
			BooleanLiteral booleanLiteral=(BooleanLiteral)node;
			String value=booleanLiteral.toString();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==InfixExpression.class)
		{
			InfixExpression infixExpression=(InfixExpression)node;
			String value=infixExpression.getOperator().toString();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==PrefixExpression.class)
		{
			PrefixExpression prefixExpression=(PrefixExpression)node;
			String value=prefixExpression.getOperator().toString();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==PostfixExpression.class)
		{
			PostfixExpression postfixExpression=(PostfixExpression)node;
			String value=postfixExpression.getOperator().toString();
			System.out.print("("+type+"("+value+")");
		}
		else if(node.nodeClassForType(node.getNodeType())==PrimitiveType.class)
		{
			PrimitiveType primitiveType=(PrimitiveType)node;
			String value=primitiveType.getPrimitiveTypeCode().toString();
			System.out.print("("+type+"("+value+")");
		}
		else System.out.print("("+type);
	}
	
	public void postVisit(ASTNode node) {
		System.out.print(")");
	}
}
