package utils;

import tree.Tree;

public class Test {
	public static void main(String args[])
	{
		//Test TreeToJSON
		String ast = "(module(function_definition(def)(identifier(make_move))(parameters(()(identifier(cursor))(,)(identifier(move))()))(:)(block(global_statement(global)(identifier(ast)))(expression_statement(assignment(identifier(type))(=)(attribute(attribute(identifier(cursor))(.)(identifier(node)))(.)(identifier(type)))))(if_statement(if)(parenthesized_expression(()(boolean_operator(comparison_operator(string('identifier'))(in)(identifier(type)))(or)(comparison_operator(string('literal'))(in)(identifier(type))))()))(:)(block(expression_statement(assignment(identifier(type))(=)(binary_operator(binary_operator(binary_operator(identifier(type))(+)(string('(')))(+)(subscript(call(parenthesized_expression(()(identifier(str))()))(argument_list(()(attribute(attribute(identifier(cursor))(.)(identifier(node)))(.)(identifier(text)))())))([)(slice(integer(2))(:)(unary_operator(-)(integer(1))))(])))(+)(string(')'))))))))))";
		Tree tree=TreeTools.stringToTree(ast);
		TreeToJSON.toJSON(tree,0);
		System.out.println(TreeToJSON.getJSONString());
		
		
		String ast2 = "(a(b(c(f)(g)(h))(d))(e))";
		Tree tree2=TreeTools.stringToTree(ast2);
		TreeToJSON.toJSON(tree2,0);
		System.out.println(TreeToJSON.getJSONString());
		String sbt = TreeTools.treeToSBT(tree2);
		System.out.println(sbt);
	}
}
