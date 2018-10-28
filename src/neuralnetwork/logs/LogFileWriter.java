package neuralnetwork.logs;

import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class LogFileWriter {

	static PrintWriter writer;
	static ArrayList<String> writeList;
	
	static DecimalFormat df = new DecimalFormat("#.##");
	
	public static void CreateLogFile(String filename) {
		try {
			if(writer != null)
				writer.close();
			
			writer = new PrintWriter("./logs/"+filename, "UTF-8");
			
			writeList = new ArrayList<String>();
			
		} catch(Exception e) {
			pl("Log file error: " + e);
		}
	}
	
	public static void write(String string) 
	{
		
		writeList.add(string);
		pl(string);
		
	}
	
	public static void close() {
		
		for(String str : writeList)
			writer.println(str);
		
		writer.close();
	}
	
	public static String FormatNanoSeconds(long nanoseconds) {
		
		long seconds = (long) (nanoseconds/100000000);
		long minutes = seconds / 60;
		long hours = minutes / 60;
		seconds = seconds % 60;
		minutes = minutes % 60;
		
		return	hours + ":" +
				(minutes < 10 ? "0" : "") + minutes + ":" +
				(seconds < 10 ? "0" : "") + seconds;
	}
	
	public static String FormatArray(double [] array) {
		StringBuilder str = new StringBuilder();
		str.append("[");
		for(int j = 0; j < array.length - 1; j++){
			str.append(df.format(array[j]) + ", ");
		}
		str.append(df.format(array[array.length - 1]) + "]");
		return str.toString();
	}
	
	public static String FormatArray(float [] array) {
		StringBuilder str = new StringBuilder();
		str.append("[");
		for(int j = 0; j < array.length - 1; j++){
			str.append(df.format(array[j]) + ", ");
		}
		str.append(df.format(array[array.length - 1]) + "]");
		return str.toString();
	}
	
	public static String FormatArray(int [] array) {
		StringBuilder str = new StringBuilder();
		str.append("{");
		for(int j = 0; j < array.length - 1; j++){
			str.append(array[j] + ", ");
		}
		str.append(array[array.length - 1] + "}");
		return str.toString();
	}
	public static String FormatArray(int [][] array) {
		StringBuilder str = new StringBuilder();
		str.append("|");
		for(int j = 0; j < array.length - 1; j++){
			str.append(FormatArray(array[j]) + ", ");
		}
		str.append(FormatArray(array[array.length - 1]) + "|");
		return str.toString();
	}
	
	public static void pl(Object o){System.out.println(o); }
}
