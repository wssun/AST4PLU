package process;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import JDT.GenerateAST;
import tree.BinaryTree;
import tree.Tree;
import utils.BinaryToJSON;
import utils.TreeTools;

public class BinaryTree_for_csn {
	private static String type="valid";
	private static String FILE_PATH="D:\\ast_dataset\\csn\\rq4\\ast\\"+type+".jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\csn\\rq4\\binary_tree\\"+type+".jsonl";
	
	public static void main(String[] args) throws IOException {
		int cnt=0;
        FileReader fr = null;
        BufferedReader br = null;
        
        File jsonFile = null;
        FileWriter fileWriter = null;
        BufferedWriter bw = null;
        try {
            fr = new FileReader(FILE_PATH);
            br = new BufferedReader(fr);
            
            jsonFile = new File(JSON_FILE_PATH);
	        if (!jsonFile.exists()) {
				jsonFile.createNewFile();	
	        }
	        fileWriter = new FileWriter(jsonFile.getAbsoluteFile());
			bw = new BufferedWriter(fileWriter);
            
            String line = "";
            //读取每一行的数据
            while ( (line = br.readLine()) != null) {
            	cnt = cnt + 1;
            	System.out.println(cnt);
            	
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
                
                Boolean success=true;
                String ast_seq="";
                try {
                	ast_seq=GenerateAST.getAST(code);
                }
                catch(Exception e){
                	success=false;
                	System.out.println(cnt+" failed");
                }
                
                if(ast_seq.length()==0||success==false)continue;
            	
                Tree ast=TreeTools.stringToTreeJDT(ast_seq);
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
	            
	            tr.put("ast",tree);
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
	}
}
