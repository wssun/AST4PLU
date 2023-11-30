package process;

import de.hunsicker.jalopy.Jalopy;


import java.io.File;

public class JavaFormat {
    public static void main(String[] args) {
    	String file_path="D:\\ast_dataset\\bcb\\split_ast\\files\\";
    	File[] files = new File(file_path).listFiles();
    	int cnt = 0;
    	for (File f : files){
            try {
                Jalopy j = new Jalopy();
                j.setEncoding("utf-8");
                j.setInput(f);
                j.setOutput(f);
                j.format();
                ++cnt;
                System.out.println(cnt);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}