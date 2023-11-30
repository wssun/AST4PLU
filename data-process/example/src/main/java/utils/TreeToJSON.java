package utils;

import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONException;
import com.alibaba.fastjson.JSONObject;

import tree.Tree;

public class TreeToJSON {
	//添加fastjson依赖
	private static JSONArray nodes;
	private static Integer globalIndex;
	
	public static void toJSON(Tree tree,int startIndex)
	{
		nodes=new JSONArray();
		globalIndex=startIndex;
		addTree(tree);
	}
	
	private static Integer addTree(Tree tree)
	{
		Integer currentIndex=globalIndex;
		++globalIndex;
		
		List<Tree> children=tree.getChildren();
		List<Integer> childrenIndex=new ArrayList<Integer>();
		if(children.size()!=0)
		{
			for(int i=0;i<children.size();++i)
			{
				childrenIndex.add(addTree(children.get(i)));
			}
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
