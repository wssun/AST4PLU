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
import tree.Test;

public class SplitAST_for_bcb {
	private static String AST_FILE_PATH="D:\\ast_dataset\\bcb\\split_ast\\data_ast.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\bcb\\split_ast\\data.jsonl";
	
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
                String idx=lineJson.getString("idx");
                JSONArray asts=lineJson.getJSONArray("asts");
                List<String> ast_seqs = JSONObject.parseArray(asts.toJSONString(),String.class);
                
                int sz=ast_seqs.size();
                JSONArray new_asts=new JSONArray();
                for(int i=0;i<sz;++i)
                {
                    Tree ast=TreeTools.stringToTree(ast_seqs.get(i));
                    Test.printTree(ast, 0);
	                TreeToJSON.toJSON(ast,0);
	                JSONArray tree=TreeToJSON.getJSONArray();
	                new_asts.add(tree);
                }
                
                JSONObject tr = new JSONObject();
	            tr.put("idx",idx);
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
