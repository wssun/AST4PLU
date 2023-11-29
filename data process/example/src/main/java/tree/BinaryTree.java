package tree;

import java.util.ArrayList;
import java.util.List;

public class BinaryTree {
	
	private String root;
	private BinaryTree left,right;
	
	public BinaryTree()
	{
		root="";
		left=null;
		right=null;
	}
	
	public String getRoot() {
		return root;
	}

	public void setRoot(String root) {
		this.root = root;
	}
	
	public BinaryTree getLeftChild() {
		return left;
	}

	public void setLeftChild(BinaryTree child) {
		this.left = child;
	}
	
	public BinaryTree getRightChild() {
		return right;
	}

	public void setRightChild(BinaryTree child) {
		this.right = child;
	}
	
	public Boolean isLeaf()
	{
		if(left==null&&right==null)return true;
		else return false;
	}
}
