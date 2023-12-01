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

public class JDT_for_bcb {
	private static String FILE_PATH="D:\\ast_dataset\\bcb\\func\\data.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\bcb\\bfs\\data.jsonl";
	
	public static void main(String[] args) throws IOException {
		int cnt = 0;
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
                JSONObject lineJson = JSONObject.parseObject(line);
                String idx=lineJson.getString("idx");
                String code=lineJson.getString("func");
                
                Boolean success=true;
                String ast_seq="";
                try {
                	ast_seq=GenerateAST.getAST(code);
                }
                catch(Exception e){
                	success=false;
                	System.out.println(idx+" failed");
                }
                
                if(ast_seq.length()==0||success==false)continue;

            	System.out.println(cnt);
            	
                Tree ast=TreeTools.stringToTreeJDT(ast_seq);
//                tree.Test.printTree(ast, 0);
                TreeToJSON.toJSON(ast,0);
                JSONArray tree=TreeToJSON.getJSONArray();
                String sbt=TreeTools.treeToSBT(ast);
                String bfs=TreeTools.treeToBFS(ast);
//                List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
                
                JSONObject tr = new JSONObject();
	            tr.put("idx",idx);
	            tr.put("func",code);
	            
	            tr.put("sbt",sbt);
	            tr.put("ast",tree);
	            tr.put("bfs",bfs);
//	            tr.put("labels",non_leaf);
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
