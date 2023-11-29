package process;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import com.alibaba.fastjson.JSONObject;

import JDT.GenerateAST;
import tree.Tree;
import utils.TreeTools;

public class ASTPath_for_bcb {
	private static String FILE_PATH="D:\\ast_dataset\\bcb\\func\\data.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\bcb\\ast_path\\data_12.jsonl";
	
	public static void main(String[] args) throws IOException {
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
            int cnt=1;
            while ( (line = br.readLine()) != null) {
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
                List<String> ast_path=TreeTools.getASTPath(ast, 8, 2);
                
                JSONObject tr = new JSONObject();
	            tr.put("idx",idx);
	            tr.put("ast_path",ast_path);
	            bw.write(tr.toString()+"\n");
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
