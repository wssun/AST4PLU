package process;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSONObject;

import JDT.GenerateAST;
import tree.Tree;
import utils.TreeTools;

public class JDT_for_Javamed {
	private static String FILE_PATH="D:\\ast_dataset\\java-small";    //目录
	private static String JSON_FILE_PATH="D:\\ast_dataset\\ast_path.jsonl";
	
	public static void main(String[] args) throws IOException {
        File dir = new File(FILE_PATH);
        List<File> allFileList = new ArrayList<>();
 
        // 判断文件夹是否存在
        if (!dir.exists()) {
            System.out.println("目录不存在");
            return;
        }
 
        getAllFile(dir, allFileList);
        System.out.println("该文件夹下共有" + allFileList.size() + "个文件");
        
        FileReader fr = null;
        BufferedReader br = null;
        
        File jsonFile = null;
        FileWriter fileWriter = null;
        BufferedWriter bw = null;

        jsonFile = new File(JSON_FILE_PATH);
        if (!jsonFile.exists()) {
			jsonFile.createNewFile();	
        }
        fileWriter = new FileWriter(jsonFile.getAbsoluteFile());
		bw = new BufferedWriter(fileWriter);
		
		int idx = 0;
    	for (File file : allFileList) {
    		idx = idx + 1;
            fr = new FileReader(file);
            br = new BufferedReader(fr);
            String file_name = file.getName();
            
            String code = "";
            String line = "";
            //读取每一行的数据
            while ( (line = br.readLine()) != null) {
            	code = code + line + "\n";
            }
//            System.out.println(code);
            
            Boolean success=true;
            String ast_seq="";
            try {
            	ast_seq=GenerateAST.getAST(code);
            }
            catch(Exception e){
            	success=false;
            	System.out.println(file_name+" failed");
            }
            
            if(ast_seq.length()==0||success==false)continue;
            
            System.out.println(idx);
            Tree ast=TreeTools.stringToTreeJDT(ast_seq);
            List<String> ast_path=TreeTools.getASTPath(ast, 8, 2);
            
            JSONObject tr = new JSONObject();
            tr.put("file",file_name);
            tr.put("ast_path",ast_path);
            bw.write(tr.toString()+"\n");

            br.close();
            fr.close();
        }
        bw.close();
	}
	
    public static void getAllFile(File fileInput, List<File> allFileList) {
        // 获取文件列表
        File[] fileList = fileInput.listFiles();
        assert fileList != null;
        for (File file : fileList) {
            if (file.isDirectory()) {
                // 递归处理文件夹
                getAllFile(file, allFileList);
            } else {
                // 如果是文件则将其加入到文件数组中
                allFileList.add(file);
            }
        }
    }
}
