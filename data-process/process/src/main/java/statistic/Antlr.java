package statistic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import com.alibaba.fastjson.JSONObject;

import antlr.GenerateAST;
import tree.Tree;
import utils.TreeTools;

public class Antlr {
	private static String language="java";
	private static String FILE_PATH="D:\\ast_dataset\\bcb\\func\\data.jsonl";
	private static int MAX_SIZE=10000;
	
	public static void main(String[] args) throws IOException {
		long[] tsList=new long[MAX_SIZE];
		long[] tdList=new long[MAX_SIZE];
		long[] bfList=new long[MAX_SIZE];
		long[] utpList=new long[MAX_SIZE];
		long[] utkList=new long[MAX_SIZE];
		int idx=0;
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(FILE_PATH);
            br = new BufferedReader(fr);
            
            String line = "";
            //读取每一行的数据
            while ( (line = br.readLine()) != null) {
                JSONObject lineJson = JSONObject.parseObject(line);
                String code=lineJson.getString("func");
                long id=lineJson.getLongValue("idx");
                String ast="";
                if(language=="java")ast=GenerateAST.getJavaAST(code);
//                else if(language=="python")ast=GenerateAST.getPythonAST(code);
//                else ast=GenerateAST.getCppAST(code);
                if(id%10000==0)System.out.println(ast);
                
                Tree tree=TreeTools.stringToTree(ast);
                tsList[idx]=tree.getTreeSize();
                tdList[idx]=tree.getTreeDepth();
                bfList[idx]=tree.getBF();
                utpList[idx]=tree.getUTP();
                utkList[idx]=tree.getUTK(0);
                idx++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            try {
                br.close();
                fr.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        Arrays.sort(tsList,0,idx);
        Arrays.sort(tdList,0,idx);
        Arrays.sort(bfList,0,idx);
        Arrays.sort(utpList,0,idx);
        Arrays.sort(utkList,0,idx);
        
        
        double tsSum=0,tdSum=0,bfSum=0,utpSum=0,utkSum=0;
        for(int i=0;i<idx;++i)
        {
        	tsSum+=tsList[i];
        	tdSum+=tdList[i];
        	bfSum+=bfList[i];
        	utpSum+=utpList[i];
        	utkSum+=utkList[i];
        }
        
        
        long tsAvg=Math.round(tsSum/idx);
        long tdAvg=Math.round(tdSum/idx);
        long bfAvg=Math.round(bfSum/idx);
        long utpAvg=Math.round(utpSum/idx);
        long utkAvg=Math.round(utkSum/idx);
        
        System.out.println("Language: "+language+", Tool: Antlr, Sum:"+idx);
        System.out.println("\t\tMin\tAverage\tMedian\tMax");
        System.out.println("Tree Size:\t"+tsList[0]+"\t"+tsAvg+"\t"+tsList[idx/2]+"\t"+tsList[idx-1]);
        System.out.println("Tree Depth:\t"+tdList[0]+"\t"+tdAvg+"\t"+tdList[idx/2]+"\t"+tdList[idx-1]);
        System.out.println("Branch Factor:\t"+bfList[0]+"\t"+bfAvg+"\t"+bfList[idx/2]+"\t"+bfList[idx-1]);
        System.out.println("Unique Types:\t"+utpList[0]+"\t"+utpAvg+"\t"+utpList[idx/2]+"\t"+utpList[idx-1]);
        System.out.println("Unique Tokens:\t"+utkList[0]+"\t"+utkAvg+"\t"+utkList[idx/2]+"\t"+utkList[idx-1]);

	}
}
