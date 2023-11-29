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

import antlr.GenerateAST;
import tree.BinaryTree;
import tree.Tree;
import utils.BinaryToJSON;
import utils.TreeToJSON;
import utils.TreeTools;

public class Antlr_for_bcb {
	private static String lang="java";
	private static String FILE_PATH="D:\\ast_dataset\\bcb\\func\\data.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\bcb\\binary_tree\\antlr\\test11.jsonl";
	
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
                JSONObject lineJson = JSONObject.parseObject(line);
                int idx=lineJson.getInteger("idx");
                String code=lineJson.getString("func");
                if(idx == 10000832 || idx == 21571495)continue;
                
                Boolean success=true;
                String ast_seq="";
                try {
                	if(lang=="java")ast_seq=GenerateAST.getJavaAST(code);
//                	else if(lang=="python")ast_seq=GenerateAST.getPythonAST(code);
//                	else if(lang=="cpp")ast_seq=GenerateAST.getCppAST(code);
//                	System.out.println(ast_seq);
                }
                catch(Exception e){
                	success=false;
//                	System.out.println(idx);
                }
                
                if(ast_seq.length()==0||success==false)continue;
                
            	System.out.println(cnt);
            	
                Tree ast=TreeTools.stringToTree(ast_seq);
//                TreeToJSON.toJSON(ast,0);
//                JSONArray tree=TreeToJSON.getJSONArray();
//                String sbt=TreeTools.treeToSBT(ast);
//                List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
                BinaryTree bn=TreeTools.TreeToBinary(ast);
                BinaryToJSON.toJSON(bn,0);
                JSONArray tree=BinaryToJSON.getJSONArray();
                
                JSONObject tr = new JSONObject();
	            tr.put("idx",Integer.toString(idx));
	            
//	            tr.put("sbt",sbt);
	            tr.put("ast",tree);
//	            tr.put("labels",non_leaf);
	            bw.write(tr.toString()+"\n");
	            ++cnt;
	            System.out.println(cnt);
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
