package utils;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import tree.BinaryTree;
import tree.Tree;

public class TreeTools {
	
	private static String sbt;
	private static List<String> sbt_no_brk;
	private static List<String> sbt_brk;
	private static List<String> non_leaf;
	private static Queue<Tree> que;
	private static List<String> ast_path;
	
	public static Tree stringToTree(String ast)
	{
		return buildTree(ast.substring(1,ast.length()-1));
	}
	
	private static Tree buildTree(String ast)
	{
		Tree tree=new Tree();
		//字符串
		if(ast.charAt(0)=='\"'||ast.charAt(0)=='\'')
		{
			tree.setRoot(ast);
			return tree;
		}
		//根节点
		int p=1;
		while(p<ast.length()&&ast.charAt(p)!='(')++p;
		String rt=ast.substring(0,p);
		if(rt.equals("<left_bracket_5175241>"))rt="(";
		if(rt.equals("<right_bracket_5175241>"))rt=")";
		tree.setRoot(rt);
		if(p>=ast.length())return tree;
		//子节点
		while(p<ast.length()&&ast.charAt(p)=='(')
		{
			//ast[p~q]表示一个子节点
			Boolean in1=false,in2=false,pre=false;
			int cnt=0,q=p;
			for(;q<ast.length();++q)
			{
				if(in1||in2)
				{
					if(ast.charAt(q)=='"')in1=false;
					if(ast.charAt(q)=='\'')in2=false;
					continue;
				}

				if(ast.charAt(q)=='"')in1=true;
				else if(ast.charAt(q)=='\'')in2=true;
				
				if(pre)
				{
					pre=false;
					continue;
				}
				
				if(ast.charAt(q)=='(')
				{
					++cnt;
					pre=true;
				}
				else if(ast.charAt(q)==')')--cnt;
				if(cnt==0)break;
			}
			tree.addChild(buildTree(ast.substring(p+1,q)));
			p=q+1;
		}
		return tree;
	}
	
	public static Tree stringToTreeJDT(String ast)
	{
		return buildTreeJDT(ast.substring(1,ast.length()-1));
	}
	
	private static Tree buildTreeJDT(String ast)
	{
		Tree tree=new Tree();
		//根节点
		int p=1;
		while(p<ast.length()&&ast.charAt(p)!='(')++p;
		tree.setRoot(ast.substring(0,p));
		if(p>=ast.length())return tree;
		//子节点
		while(p<ast.length()&&ast.charAt(p)=='(')
		{
			//ast[p~q]表示一个子节点
			int cnt=0,q=p;
			for(;q<ast.length();++q)
			{
				if(ast.charAt(q)=='(')++cnt;
				else if(ast.charAt(q)==')')--cnt;
				if(cnt==0)break;
			}
			if(p+1<q)tree.addChild(buildTreeJDT(ast.substring(p+1,q)));
			else tree.addChild(buildTreeJDT("\"\""));
			p=q+1;
		}
		return tree;
	}
	
	private static void traverse_sbt(Tree tree)
	{
		String root=tree.getRoot();
		sbt=sbt+"("+root;
		
		List<Tree> children=tree.getChildren();
		int len=children.size();
		for(int i=0;i<len;++i)
		{
			traverse_sbt(children.get(i));
		}
		
		sbt=sbt+")"+root;
	}
	
	public static String treeToSBT(Tree tree)
	{
		sbt="";
		traverse_sbt(tree);
		return sbt;
	}
	
	private static void traverse_sbt_array_no_brackets(Tree tree)
	{
		String root=tree.getRoot();
		sbt_no_brk.add(root);
		
		List<Tree> children=tree.getChildren();
		int len=children.size();
		for(int i=0;i<len;++i)
		{
			traverse_sbt_array_no_brackets(children.get(i));
		}
		
		sbt_no_brk.add(root);
	}
	
	public static List<String> treeToSBTArrayNoBrackets(Tree tree)
	{
		sbt_no_brk=new ArrayList<String>();
		traverse_sbt_array_no_brackets(tree);
		return sbt_no_brk;
	}
	
	private static void traverse_sbt_array_brackets(Tree tree)
	{
		String root=tree.getRoot();
		sbt_brk.add("(");
		sbt_brk.add(root);
		
		List<Tree> children=tree.getChildren();
		int len=children.size();
		for(int i=0;i<len;++i)
		{
			traverse_sbt_array_brackets(children.get(i));
		}
		
		sbt_brk.add(")");
		sbt_brk.add(root);
	}
	
	public static List<String> treeToSBTArrayBrackets(Tree tree)
	{
		sbt_brk=new ArrayList<String>();
		traverse_sbt_array_brackets(tree);
		return sbt_brk;
	}
	
	public static String treeToBFS(Tree tree)
	{
		String bfs="";
		Boolean fir=true;
		que=new LinkedList<Tree>();
		
		que.offer(tree);
		while(que.size()!=0)
		{
			Tree top=que.poll();
			if(!fir)bfs+=" ";
			bfs+=top.getRoot();
			fir=false;
			
			List<Tree> children=top.getChildren();
			int len=children.size();
			for(int i=0;i<len;++i)
			{
				que.offer(children.get(i));
			}
		}
		return bfs;
	}
	
	private static void traverse_non_leaf(Tree tree)
	{
		String root=tree.getRoot();
		List<Tree> children=tree.getChildren();
		if(children.size()==0)return;
		
		non_leaf.add(root);
		int len=children.size();
		for(int i=0;i<len;++i)
		{
			traverse_non_leaf(children.get(i));
		}
	}
	
	public static List<String> treeToNonLeaf(Tree tree)
	{
		non_leaf=new ArrayList<String>();
		traverse_non_leaf(tree);
		return non_leaf;
	}
	
	private static void findRightLeaf(Tree root,int len,String path)
	{
		List<Tree> children = root.getChildren();
		if(children.size()==0)
		{
			path=path+"<sep>"+root.getRoot();  //","不能作为分隔符(有values(?,?,?)这样的sql语句，分隔符换成<sep>)
			ast_path.add(path);
		}
		else
		{
			if(len<=0)return; //已经到了ast path最大长度
			int sz=children.size();
			for(int i=0;i<sz;++i)
			{
				String new_path=path+"|"+root.getRoot();
				findRightLeaf(children.get(i),len-1,new_path);
			}
		}
	}
	
	private static void findPathForLeaf(Tree leaf, int maxLen, int maxWid)
	{
		String path=leaf.getRoot()+"<sep>";
		Tree top=leaf, pre=leaf;
		int len=-1; //length of the ast path: number of nonleaf nodes - 1
		while(!top.isRoot())  //判断是否是根节点
		{
			pre=top;
			top=top.getFather();
			if(len!=-1)path+="|";
			path+=top.getRoot();
			++len;
			if(len>maxLen)break;
			
			List<Tree> children = top.getChildren();
			int sz=children.size(),j=0;
			for(;j<sz;++j)
			{
				if(children.get(j)==pre) break;
			}
			for(int i=j+1;i<sz&&i-j<=maxWid;++i)
			{
				findRightLeaf(children.get(i),maxLen-len,path);
			}
		
		}
	}
	
	private static void findLeftLeaf(Tree root,int maxLen,int maxWid)
	{
		List<Tree> children = root.getChildren();
		if(children.size()==0)
		{
			findPathForLeaf(root,maxLen,maxWid);
		}
		else
		{
			int sz=children.size();
			for(int i=0;i<sz;++i)
			{
				findLeftLeaf(children.get(i),maxLen,maxWid);
			}
		}
	}
	
	public static List<String> getASTPath(Tree tree, int maxLen, int maxWid)
	{
		ast_path=new ArrayList<String>();
		findLeftLeaf(tree, maxLen, maxWid);
		return ast_path;
	}

	
	public static BinaryTree TreeToBinary(Tree tree)
	{
		BinaryTree bt;
		List<Tree> children = tree.getChildren();
		if(children.size()==0)
		{
			bt = new BinaryTree();
			bt.setRoot(tree.getRoot());
		}
		else if(children.size()==1)bt = TreeToBinary(children.get(0));
		else if(children.size()==2)
		{
			bt = new BinaryTree();
			bt.setRoot(tree.getRoot());
			bt.setLeftChild(TreeToBinary(children.get(0)));
			bt.setRightChild(TreeToBinary(children.get(1)));
		}
		else  // children.size() > 2
		{
			bt = new BinaryTree();
			bt.setRoot(tree.getRoot());
			bt.setLeftChild(TreeToBinary(children.get(0)));
			
			Tree right = new Tree();
			right.setRoot("Temp");
			for(int i=1;i<children.size();++i)right.addChild(children.get(i));
			bt.setRightChild(TreeToBinary(right));
		}
		return bt;
	}
}
