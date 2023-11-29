package JDT;

import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.BooleanLiteral;
import org.eclipse.jdt.core.dom.InfixExpression;
import org.eclipse.jdt.core.dom.Modifier;
import org.eclipse.jdt.core.dom.NumberLiteral;
import org.eclipse.jdt.core.dom.PostfixExpression;
import org.eclipse.jdt.core.dom.PrefixExpression;
import org.eclipse.jdt.core.dom.PrimitiveType;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.StringLiteral;

public class noTokenVisitor {
	public void preVisit(ASTNode node) {
		String type=node.nodeClassForType(node.getNodeType()).getSimpleName();
		System.out.print("("+type);
	}
	
	public void postVisit(ASTNode node) {
		System.out.print(")");
	}
}
