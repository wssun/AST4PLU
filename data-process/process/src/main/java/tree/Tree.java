package tree;

import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class Tree{
	private String root;
	private Tree father;
	private List<Tree> children;
	private HashSet<String> utp;
	private HashSet<String> utk;
	private long nonleaf;
	private long sum_children;
	
	public Tree()
	{
		root="";
		children=new ArrayList<Tree>();
		father=null;
	}
	
	public String getRoot() {
		return root;
	}

	public void setRoot(String root) {
		this.root = root;
	}
	
	public Boolean isRoot() {
		if(father==null)return true;
		return false;
	}
	
	public Tree getFather() {
		return father;
	}

	public void setfather(Tree fa) {
		this.father = fa;
	}
	
	public List<Tree> getChildren(){
		return children;
	}
	
	public void addChild(Tree child)
	{
		child.setfather(this);
		children.add(child);
	}
	
	public boolean isLeaf()
	{
		return children.size()==0;
	}
	
	public long getTreeSize()
	{
		if(root=="")return 0;
		
		long ts=1;
		int sz=children.size();
		for(int i=0;i<sz;++i)
		{
			ts+=children.get(i).getTreeSize();
		}
		return ts;
	}
	
	public long getTreeDepth()
	{
		if(root=="")return 0;
		
		long td=1;
		int sz=children.size();
		for(int i=0;i<sz;++i)
		{
			long tmp=children.get(i).getTreeDepth()+1;
			if(tmp>td)td=tmp;
		}
		return td;
	}
	
	//BF:branching factor — mean number of children in nonleaf vertices of a tree
	private void countBF(Tree tree)
	{
		if(tree.isLeaf())return;
		nonleaf+=1;
		
		List<Tree> c=tree.getChildren();
		int sz=c.size();
		sum_children+=sz;
		
		for(int i=0;i<sz;++i)
		{
			countBF(c.get(i));
		}
	}

	public long getBF()
	{
		nonleaf=0;
		sum_children=0;
		countBF(this);
		if(nonleaf==0)return 0;
		else return Math.round(sum_children*1.0/nonleaf);
	}
	
	//UTP:unique types — number of unique types of intermediate nodes used in an AST
	private void countUTP(Tree tree)
	{
		if(tree.isLeaf())return;
		utp.add(tree.getRoot());
		
		List<Tree> c=tree.getChildren();
		int sz=c.size();
		for(int i=0;i<sz;++i)
		{
			countUTP(c.get(i));
		}
	}

	public long getUTP()
	{
		utp=new HashSet<String>();
		countUTP(this);
		return utp.size();
	}
	
	private static String[] splitCamelCase(String input) {
		Pattern pattern = Pattern.compile("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])");
        Matcher matcher = pattern.matcher(input);
        return matcher.replaceAll(" ").split(" ");
    }
	
	private static String[] splitSnakeCase(String input) {
        return input.split("_");
    }
	
	//UTK:unique tokens — number of unique sub-tokens in AST leaves
	private void countUTK(Tree tree, int flg)
	{
		if(tree.isLeaf()) 
		{
			if(tree.getRoot()!="")
			{
				String[] subtokens;
				if(flg==0)subtokens = splitCamelCase(tree.getRoot());
				else subtokens = splitSnakeCase(tree.getRoot());
				for(int i=0;i<subtokens.length;++i) {
					utk.add(subtokens[i]);
//					System.out.println(subtokens[i]);
				}
//				System.out.println();
			}
			return;
		}
		
		List<Tree> c=tree.getChildren();
		int sz=c.size();
		for(int i=0;i<sz;++i)
		{
			countUTK(c.get(i), flg);
		}
	}

	public long getUTK(int flg)  //flg==0: camelCaseSplitting; fLg==1: snake_case_splitting
	{
		utk=new HashSet<String>();
		countUTK(this, flg);
		return utk.size();
	}
}
