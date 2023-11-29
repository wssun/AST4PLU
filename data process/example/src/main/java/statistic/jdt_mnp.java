package statistic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

import JDT.GenerateAST;
import tree.Tree;
import utils.TreeTools;


//used for method name prediction
public class jdt_mnp {
	private static String DIR_PATH="D:\\training\\";
	private static String LOG_FILE_PATH="D:\\log.txt";
	private static int MAX_SIZE=630000;
	private static String language="java";
	private static int file_num = 629762;
	
	public static void main(String[] args) throws IOException {
		long[] tsList=new long[MAX_SIZE];
		long[] tdList=new long[MAX_SIZE];
		long[] bfList=new long[MAX_SIZE];
		long[] utpList=new long[MAX_SIZE];
		long[] utkList=new long[MAX_SIZE];
		
		int idx=0;
        FileReader fr = null;
        BufferedReader br = null;
        
        File logFile = null;
        FileWriter fileWriter = null;
        BufferedWriter bw = null;
        try {
            logFile = new File(LOG_FILE_PATH);
	        if (!logFile.exists()) {
	        	logFile.createNewFile();	
	        }
	        fileWriter = new FileWriter(logFile.getAbsoluteFile());
			bw = new BufferedWriter(fileWriter);
			
        	for(int i=1;i<=file_num;++i)
        	{
        		String FILE_PATH = DIR_PATH + "training"+ i + ".jsonl";
        	
	            fr = new FileReader(FILE_PATH);
	            br = new BufferedReader(fr);
	            
	            String line = "";
	            //读取每一行的数据
	            while ( (line = br.readLine()) != null) {
	                JSONObject lineJson = JSONObject.parseObject(line);
	                String code=lineJson.getString("code");
	                String ast=GenerateAST.getAST(code);
	                if(idx%10000==0)System.out.println(ast);
	                
	                Tree tree=TreeTools.stringToTreeJDT(ast);
	                long ts=tree.getTreeSize();
	                long td=tree.getTreeDepth();
	                long bf=tree.getBF();
	                long utp=tree.getUTP();
	                long utk=tree.getUTK();
	                tsList[idx]=ts;
	                tdList[idx]=td;
	                bfList[idx]=bf;
	                utpList[idx]=utp;
	                utkList[idx]=utk;
		            bw.write(ts+"\t"+td+"\t"+bf+"\t"+utp+"\t"+utk+"\t");
	                idx++;
	            }
	            
	            try {
	                br.close();
	                fr.close();
	            } catch (Exception e) {
	                e.printStackTrace();
	            }
        	}
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            try {
            	bw.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("total:"+idx);
		
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
        
        System.out.println("Language: "+language+", Tool: JDT, Sum:"+idx);
        System.out.println("\t\tMin\tAverage\tMedian\tMax");
        System.out.println("Tree Size:\t"+tsList[0]+"\t"+tsAvg+"\t"+tsList[idx/2]+"\t"+tsList[idx-1]);
        System.out.println("Tree Depth:\t"+tdList[0]+"\t"+tdAvg+"\t"+tdList[idx/2]+"\t"+tdList[idx-1]);
        System.out.println("Branch Factor:\t"+bfList[0]+"\t"+bfAvg+"\t"+bfList[idx/2]+"\t"+bfList[idx-1]);
        System.out.println("Unique Types:\t"+utpList[0]+"\t"+utpAvg+"\t"+utpList[idx/2]+"\t"+utpList[idx-1]);
        System.out.println("Unique Tokens:\t"+utkList[0]+"\t"+utkAvg+"\t"+utkList[idx/2]+"\t"+utkList[idx-1]);

	}

}
