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

import JDT.GenerateAST;
import tree.Tree;
import utils.TreeToJSON;
import utils.TreeTools;

public class SplitAST_for_csn {
	private static String AST_FILE_PATH="D:\\ast_dataset\\csn\\split_ast\\train_ast.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\csn\\split_ast\\train.jsonl";
	
	// use Tree-sitter
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
            int cnt=1;
            while ( (line = br.readLine()) != null) {
                JSONObject lineJson = JSONObject.parseObject(line);
                String repo=lineJson.getString("repo");
                String path=lineJson.getString("path");
                String func_name=lineJson.getString("func_name");
                String original_string=lineJson.getString("original_string");
                String language=lineJson.getString("language");
                String original_code=lineJson.getString("code");
                JSONArray code_tokens=lineJson.getJSONArray("code_tokens");
                String docstring=lineJson.getString("docstring");
                JSONArray docstring_tokens=lineJson.getJSONArray("docstring_tokens");
                JSONArray asts=lineJson.getJSONArray("asts");
                List<String> ast_seqs = JSONObject.parseArray(asts.toJSONString(),String.class);
                
                int sz=ast_seqs.size();
                JSONArray new_asts=new JSONArray();
                for(int i=0;i<sz;++i)
                {
                    Tree ast=TreeTools.stringToTree(ast_seqs.get(i));
	                TreeToJSON.toJSON(ast,0);
	                JSONArray tree=TreeToJSON.getJSONArray();
	                new_asts.add(tree);
                }
                
                JSONObject tr = new JSONObject();
	            tr.put("repo",repo);
	            tr.put("path",path);
	            tr.put("func_name",func_name);
	            tr.put("original_string",original_string);
	            tr.put("language",language);
	            tr.put("code",original_code);
	            tr.put("code_tokens",code_tokens);
	            tr.put("docstring",docstring);
	            tr.put("docstring_tokens",docstring_tokens);
	            tr.put("asts",new_asts);
	            bw.write(tr.toString()+"\n");
	            System.out.println(cnt);
	            ++cnt;
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
