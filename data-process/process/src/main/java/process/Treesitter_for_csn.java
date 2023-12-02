package process;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import tree.BinaryTree;
import tree.Tree;
import utils.BinaryToJSON;
import utils.TreeToJSON;
import utils.TreeTools;

public class Treesitter_for_csn {
	private static String type="valid";
	private static String AST_FILE_PATH="D:\\ast_dataset\\csn\\rq4\\ast\\"+type+"_ast.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\csn\\rq4\\ast\\"+type+".jsonl";
	
	public static void main(String[] args) throws IOException {
        FileReader fr = null;
        BufferedReader br = null;
        
        File jsonFile = null;
        FileWriter fileWriter = null;
        BufferedWriter bw = null;
        try {
            fr = new FileReader(AST_FILE_PATH);
            br = new BufferedReader(fr);
            
            jsonFile = new File(JSON_FILE_PATH);
	        if (!jsonFile.exists()) {
				jsonFile.createNewFile();	
	        }
	        fileWriter = new FileWriter(jsonFile.getAbsoluteFile());
			bw = new BufferedWriter(fileWriter);
            
            String line = "";
            //读取每一行的数据
            int idx=0;
            while ( (line = br.readLine()) != null) {
            	idx++;
            	System.out.println(idx);
            	
                JSONObject lineJson = JSONObject.parseObject(line);
                String repo=lineJson.getString("repo");
                String path=lineJson.getString("path");
                String func_name=lineJson.getString("func_name");
                String original_string=lineJson.getString("original_string");
                String language=lineJson.getString("language");
                String code=lineJson.getString("code");
                JSONArray code_tokens=lineJson.getJSONArray("code_tokens");
                String docstring=lineJson.getString("docstring");
                JSONArray docstring_tokens=lineJson.getJSONArray("docstring_tokens");
                String ast_seq=lineJson.getString("ast");
            	
                Tree ast=TreeTools.stringToTree(ast_seq);
//              TreeToJSON.toJSON(ast,0);
//              JSONArray tree=TreeToJSON.getJSONArray();
//              List<String> sbt=TreeTools.treeToSBTArrayBrackets(ast);
//              List<String> nonleaf=TreeTools.treeToNonLeaf(ast);
       		    BinaryTree bn=TreeTools.TreeToBinary(ast);
                BinaryToJSON.toJSON(bn,0);
                JSONArray tree=BinaryToJSON.getJSONArray();
                
                JSONObject tr = new JSONObject();
	            tr.put("repo",repo);
	            tr.put("path",path);
	            tr.put("func_name",func_name);
	            tr.put("original_string",original_string);
	            tr.put("language",language);
	            tr.put("code",code);
	            tr.put("code_tokens",code_tokens);
	            tr.put("docstring",docstring);
	            tr.put("docstring_tokens",docstring_tokens);
	            
	            tr.put("sbt",sbt);
	            tr.put("ast",tree);
//                tr.put("labels",nonleaf);
	            bw.write(tr.toString()+"\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            try {
            	bw.close();
                br.close();
                fr.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("finish");
	}
}
