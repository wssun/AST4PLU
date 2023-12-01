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

public class SrcML_for_csn {
	private static String lang="java";
	private static String type="train";
	private static String FILE_PATH="D:\\ast_dataset\\csn\\original\\"+type+".jsonl";
	private static String JSON_FILE_PATH="D:\\ast_dataset\\csn\\"+type+"_srcml_nonleaf.jsonl";
	private static String TEMP_CODE_PATH="D:\\ast_dataset\\csn\\temp";
	private static String TEMP_AST_PATH="D:\\ast_dataset\\csn\\ast.xml";
	
	public static void main(String[] args) throws IOException {
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
            int idx=0;
            while ( (line = br.readLine()) != null) {
            	idx++;
            	System.out.println(idx);
            	
                JSONObject lineJson = JSONObject.parseObject(line);
//                String repo=lineJson.getString("repo");
//                String path=lineJson.getString("path");
//                String func_name=lineJson.getString("func_name");
//                String original_string=lineJson.getString("original_string");
//                String language=lineJson.getString("language");
                String code=lineJson.getString("code");
//                JSONArray code_tokens=lineJson.getJSONArray("code_tokens");
//                String docstring=lineJson.getString("docstring");
//                JSONArray docstring_tokens=lineJson.getJSONArray("docstring_tokens");

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
//                TreeToJSON.toJSON(ast,0);
//                JSONArray tree=TreeToJSON.getJSONArray();
//                List<String> sbt=TreeTools.treeToSBTArrayBrackets(ast);
                List<String> non_leaf=TreeTools.treeToNonLeaf(ast);
                
                JSONObject tr = new JSONObject();
//	            tr.put("repo",repo);
//	            tr.put("path",path);
//	            tr.put("func_name",func_name);
//	            tr.put("original_string",original_string);
//	            tr.put("language",language);
//	            tr.put("code",code);
//	            tr.put("code_tokens",code_tokens);
//	            tr.put("docstring",docstring);
//	            tr.put("docstring_tokens",docstring_tokens);
//	            
//	            tr.put("sbt",sbt);
//	            tr.put("ast",tree);
                tr.put("labels",non_leaf);
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
