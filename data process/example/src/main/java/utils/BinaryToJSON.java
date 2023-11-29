package utils;

import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import tree.BinaryTree;
import tree.Tree;

public class BinaryToJSON {
	//添加fastjson依赖
	private static JSONArray nodes;
	private static Integer globalIndex;
	
	public static void toJSON(BinaryTree tree,int startIndex)
	{
		nodes=new JSONArray();
		globalIndex=startIndex;
		addTree(tree);
	}
	
	private static Integer addTree(BinaryTree tree)
	{
		Integer currentIndex=globalIndex;
		++globalIndex;
		
		List<Integer> childrenIndex=new ArrayList<Integer>();
		if(!tree.isLeaf())
		{
			childrenIndex.add(addTree(tree.getLeftChild()));
			childrenIndex.add(addTree(tree.getRightChild()));
		}
		else
		{
			childrenIndex.add(-1);
		}
		
		
		JSONObject root = new JSONObject();
		try {
            root.put("id", currentIndex);
            root.put("label", tree.getRoot());
            root.put("children", childrenIndex);
            nodes.add(root);
        } catch (Exception e) {
            e.printStackTrace();
        }
		
		return currentIndex;
	}
	
	public static JSONArray getJSONArray()
	{
		return nodes;
	}
	
	public static String getJSONString()
	{
		return nodes.toJSONString();
	}
}
