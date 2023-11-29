 package tree;

import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSONArray;

import JDT.GenerateAST;
import utils.TreeToJSON;
import utils.TreeTools;
import utils.XmlToAST;

public class Test {
	
	public static void main(String[] args) {
		//ast需要至少包含一个节点
		//string_literal的子节点包含\n的话要加上转义符（如“%d\n”→“%d\\n”），不然printTree的时候会作为换行符输出

		String t1="    private String GetResponse(URL url) {\r\n" + 
				"        String content = null;\r\n" + 
				"        try {\r\n" + 
				"            HttpURLConnection conn = (HttpURLConnection) url.openConnection();\r\n" + 
				"            conn.setDoOutput(false);\r\n" + 
				"            conn.setRequestMethod(\"GET\");\r\n" + 
				"            if (conn.getResponseCode() == HttpURLConnection.HTTP_OK) {\r\n" + 
				"                BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()));\r\n" + 
				"                String line;\r\n" + 
				"                while ((line = br.readLine()) != null) content += line;\r\n" + 
				"            } else {\r\n" + 
				"            }\r\n" + 
				"        } catch (MalformedURLException e) {\r\n" + 
				"            e.getStackTrace();\r\n" + 
				"        } catch (IOException e) {\r\n" + 
				"            e.getStackTrace();\r\n" + 
				"        }\r\n" + 
				"        return content;\r\n" + 
				"    }";
		String t2="    private String postXml(String url, String soapAction, String xml) {\r\n" + 
				"        try {\r\n" + 
				"            URLConnection conn = new URL(url).openConnection();\r\n" + 
				"            if (conn instanceof HttpURLConnection) {\r\n" + 
				"                HttpURLConnection hConn = (HttpURLConnection) conn;\r\n" + 
				"                hConn.setRequestMethod(\"POST\");\r\n" + 
				"            }\r\n" + 
				"            conn.setConnectTimeout(this.connectionTimeout);\r\n" + 
				"            conn.setReadTimeout(this.connectionTimeout);\r\n" + 
				"            conn.setRequestProperty(\"Content-Type\", \"text/xml; charset=utf-8\");\r\n" + 
				"            conn.setRequestProperty(\"Accept\", \"application/soap+xml, text/*\");\r\n" + 
				"            if (soapAction != null) {\r\n" + 
				"                conn.setRequestProperty(\"SOAPAction\", soapAction);\r\n" + 
				"            }\r\n" + 
				"            conn.setDoOutput(true);\r\n" + 
				"            OutputStreamWriter out = new OutputStreamWriter(conn.getOutputStream());\r\n" + 
				"            out.write(xml);\r\n" + 
				"            out.close();\r\n" + 
				"            BufferedReader resp = new BufferedReader(new InputStreamReader(conn.getInputStream()));\r\n" + 
				"            StringBuilder buf = new StringBuilder();\r\n" + 
				"            String str;\r\n" + 
				"            while ((str = resp.readLine()) != null) {\r\n" + 
				"                buf.append(str);\r\n" + 
				"            }\r\n" + 
				"            return buf.toString();\r\n" + 
				"        } catch (IOException e) {\r\n" + 
				"            throw new RuntimeException(e);\r\n" + 
				"        }\r\n" + 
				"    }";
		
		String s1="    @Override\r\n" + 
				"    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {\r\n" + 
				"        String rewrittenQueryString = URLDecoder.decode(request.getRequestURI(), \"UTF-8\").replaceFirst(\"^.*?\\\\/(id:.*)\\\\/.*?$\", \"$1\");\r\n" + 
				"        logger.debug(\"rewrittenQueryString: \" + rewrittenQueryString);\r\n" + 
				"        URL rewrittenUrl = new URL(fedoraUrl + rewrittenQueryString);\r\n" + 
				"        logger.debug(\"rewrittenUrl: \" + rewrittenUrl.getProtocol() + \"://\" + rewrittenUrl.getHost() + \":\" + rewrittenUrl.getPort() + rewrittenUrl.getFile());\r\n" + 
				"        HttpURLConnection httpURLConnection = (HttpURLConnection) rewrittenUrl.openConnection();\r\n" + 
				"        HttpURLConnection.setFollowRedirects(false);\r\n" + 
				"        httpURLConnection.connect();\r\n" + 
				"        response.setStatus(httpURLConnection.getResponseCode());\r\n" + 
				"        logger.debug(\"[status=\" + httpURLConnection.getResponseCode() + \"]\");\r\n" + 
				"        logger.debug(\"[headers]\");\r\n" + 
				"        for (Entry<String, List<String>> header : httpURLConnection.getHeaderFields().entrySet()) {\r\n" + 
				"            if (header.getKey() != null) {\r\n" + 
				"                for (String value : header.getValue()) {\r\n" + 
				"                    if (value != null) {\r\n" + 
				"                        logger.debug(header.getKey() + \": \" + value);\r\n" + 
				"                        if (!header.getKey().equals(\"Server\") && !header.getKey().equals(\"Transfer-Encoding\")) {\r\n" + 
				"                            response.addHeader(header.getKey(), value);\r\n" + 
				"                        }\r\n" + 
				"                    }\r\n" + 
				"                }\r\n" + 
				"            }\r\n" + 
				"        }\r\n" + 
				"        logger.debug(\"[/headers]\");\r\n" + 
				"        InputStream inputStream = httpURLConnection.getInputStream();\r\n" + 
				"        OutputStream outputStream = response.getOutputStream();\r\n" + 
				"        IOUtils.copy(inputStream, outputStream);\r\n" + 
				"    }";
		String s2="    public static void copyFile(File in, File out) throws Exception {\r\n" + 
				"        FileChannel sourceChannel = new FileInputStream(in).getChannel();\r\n" + 
				"        FileChannel destinationChannel = new FileOutputStream(out).getChannel();\r\n" + 
				"        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);\r\n" + 
				"        sourceChannel.close();\r\n" + 
				"        destinationChannel.close();\r\n" + 
				"    }";
				
		String code="    private String postXml() {\r\n" + 
				"        try {\r\n" + 
				"            URLConnection conn = new URL(url).openConnection();\r\n" + 
				"        } catch (IOException e) {\r\n" + 
				"        }\r\n" + 
				"    }\r\n";
//		String ast=GenerateAST.getAST(code);
//        System.out.println(ast);
        String ast=GenerateAST.getMaskedAST(code);
//        
        System.out.println(ast);
//		String ast = "(xaJasJds(AsdXasd)(Asd_acda)(sdh_asj(jjsSDca(nsjxSIJKxs))(ssx_xsxs)))";

		//		String ast="(1(2(5)(6)(7(8)))(3(9)(10))(4))";
//		String ast=XmlToAST.getAST("D:\\code.xml");
//		System.out.println(ast);
		
		
		Tree tree=TreeTools.stringToTree(ast);
//		Tree tree=TreeTools.stringToTreeJDT(ast);
//		String sbt=TreeTools.treeToSBT(tree);
//		System.out.println(sbt);
//        TreeToJSON.toJSON(tree,0);
//        JSONArray tree_json=TreeToJSON.getJSONArray();
//        System.out.println(tree_json);
//		List<String> paths=TreeTools.getASTPath(tree,8,2);
//		System.out.println(paths);
//		String bfs=TreeTools.treeToBFS(tree);
//		System.out.println(bfs);
//		List<String> non_leaf=TreeTools.treeToNonLeaf(tree);
//		System.out.println(non_leaf);
//		System.out.println(non_leaf.size());
		printTree(tree,0);
		
//		BinaryTree bn=TreeTools.TreeToBinary(tree);
//		printBinaryTree(bn,0);
//		
//		System.out.println("Tree Size:"+tree.getTreeSize());
//		System.out.println("Tree Depth:"+tree.getTreeDepth());
//		System.out.println("Branch Factor:"+tree.getBF());
//		System.out.println("Unique Types:"+tree.getUTP());
		System.out.println("Unique Tokens:"+tree.getUTK(0));
		
	}

	public static Tree buildTree(String ast)
	{
		Tree tree=new Tree();
		//字符串
		if(ast.charAt(0)=='\"'||ast.charAt(0)=='\'')
		{
			tree.setRoot(ast);
			return tree;
		}
		//根节点
		int p=1;
		while(p<ast.length()&&ast.charAt(p)!='(')++p;
		tree.setRoot(ast.substring(0,p));
		if(p>=ast.length())return tree;
		//子节点
		while(p<ast.length()&&ast.charAt(p)=='(')
		{
			//ast[p~q]表示一个子节点
			Boolean in1=false,in2=false,pre=false;
			int cnt=0,q=p;
			for(;q<ast.length();++q)
			{
				if(in1||in2)
				{
					if(ast.charAt(q)=='"')in1=false;
					if(ast.charAt(q)=='\'')in2=false;
					continue;
				}

				if(ast.charAt(q)=='"')in1=true;
				else if(ast.charAt(q)=='\'')in2=true;
				
				if(pre)
				{
					pre=false;
					continue;
				}
				
				if(ast.charAt(q)=='(')
				{
					++cnt;
					pre=true;
				}
				else if(ast.charAt(q)==')')--cnt;
				if(cnt==0)break;
			}
			tree.addChild(buildTree(ast.substring(p+1,q)));
			p=q+1;
		}
		return tree;
	}

	public static BinaryTree convertTreeToBinaryTree(Tree tree)
	{
		BinaryTree bn=new BinaryTree();
		
		List<Tree> children=tree.getChildren();
		if(children.size()==0)bn.setRoot(tree.getRoot());
		else if(children.size()==1)
		{
			bn=convertTreeToBinaryTree(children.get(0));
		}
		else if(children.size()==2)
		{
			bn.setRoot(tree.getRoot());
			BinaryTree left=convertTreeToBinaryTree(children.get(0));
			BinaryTree right=convertTreeToBinaryTree(children.get(1));
			bn.setLeftChild(left);
			bn.setRightChild(right);
		}
		else
		{
			bn.setRoot(tree.getRoot());
			BinaryTree left=convertTreeToBinaryTree(children.get(0));
			bn.setLeftChild(left);
			
			Tree temp=new Tree();
			temp.setRoot("Temp");
			for(int i=1;i<children.size();++i)temp.addChild(children.get(i));
			BinaryTree right=convertTreeToBinaryTree(temp);
			bn.setRightChild(right);
		}
		
		return bn;
	}
	
	public static void printTree(Tree tree,int cnt)
	{
		for(int i=0;i<cnt;++i)System.out.print("        ");
		System.out.println(tree.getRoot());
		
		List<Tree> children=tree.getChildren();
		for(int i=0;i<children.size();++i)
		{
			printTree(children.get(i),cnt+1);
		}
	}
	
	public static void printBinaryTree(BinaryTree tree,int cnt)
	{
		for(int i=0;i<cnt;++i)System.out.print("        ");
		System.out.println(tree.getRoot());
		
		if(tree.isLeaf())return;
		
		printBinaryTree(tree.getLeftChild(),cnt+1);
		printBinaryTree(tree.getRightChild(),cnt+1);
	}
}
