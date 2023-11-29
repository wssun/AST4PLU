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

import tree.Tree;
import utils.TreeToJSON;
import utils.TreeTools;
import utils.XmlToAST;

public class SrcML_for_bcb {
	private static String lang="java";
	private static String FILE_PATH="D:\\ast_dataset\\bcb\\func\\data.jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\bcb\\ast\\srcml\\data.jsonl";
	private static String TEMP_CODE_PATH="D:\\ast_dataset\\bcb\\ast\\srcml\\temp";
	private static String TEMP_AST_PATH="D:\\ast_dataset\\bcb\\ast\\srcml\\ast.xml";
	
	public static void main(String[] args) throws IOException {
		int cnt = 0;
		if(lang=="java")TEMP_CODE_PATH=TEMP_CODE_PATH+".java";
		else if(lang=="cpp")TEMP_CODE_PATH=TEMP_CODE_PATH+".cpp";
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

                File codeFile = null;
                FileWriter codeWriter = null;
                BufferedWriter codebw = null;
    			codeFile = new File(TEMP_CODE_PATH);
    	        if (!codeFile.exists()) {
    	        	codeFile.createNewFile();	
    	        }
    	        codeWriter = new FileWriter(codeFile.getAbsoluteFile());
    	        codebw = new BufferedWriter(codeWriter);
                codebw.write(code);
	            codebw.close();
	            
                String cmd = "srcml "+TEMP_CODE_PATH+" -o "+TEMP_AST_PATH;
                Process proc = Runtime.getRuntime().exec(cmd);
                proc.waitFor();
                if(proc!=null){
                    proc.destroy();
                }
                String ast_seq=XmlToAST.getAST(TEMP_AST_PATH);
                
                Tree ast=TreeTools.stringToTree(ast_seq);
                TreeToJSON.toJSON(ast,0);
                JSONArray tree=TreeToJSON.getJSONArray();
//                String sbt=TreeTools.treeToSBT(ast);
//                List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
                
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
