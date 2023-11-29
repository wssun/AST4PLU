package utils;

import java.io.File;
import java.util.List;

import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

public class XmlToAST {
	private static String ast;
	
	//path="C:\\Users\\水穷处云起时\\Desktop\\AST\\Test.xml"
	public static String getAST(String path)
	{
		ast="";
		try {
            SAXReader reader = new SAXReader();
            Document document = reader.read(new File(path));
            Element rootElement = document.getRootElement();
//            List<Element> listElement=rootElement.elements();
////            Element element =rootElement.element("function");
//            if(listElement.size()!=1)System.out.println("Warning: listElement.size()!=1 !");
//            Element element = listElement.get(0);
            traverseAST(rootElement);

        } catch (DocumentException e) {
            e.printStackTrace();
        }
		return ast;
	}
	
	public static void traverseAST(Element root)
	{
//		String rt = root.getName();
//		String txt = root.getText();
		ast = ast + "(" + root.getName();
		List<Element> elements =root.elements();
		if(elements.size()==0)
		{
			String text = root.getText();
			if(text.equals("()"))
				ast = ast + "(()())";
			else
				ast = ast + "("+text+")";
		}
		else
		{
			for(Element element:elements){
				traverseAST(element);
			}
		}
		ast = ast + ")";
	}
}
