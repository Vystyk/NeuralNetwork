// ----------------------------------------------------
// ----------------------------------------------------
// ------------------ Victor Jacobson -----------------
// ----------- simple test format with chars ----------
// ------- IDX file format for MNIST training data ----
// ----------------------------------------------------

package neuralnetwork.data;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

public class DataLoader {
	
//	public static TrainingData LoadTrainingData(String filename) 
//	{
//		BufferedReader reader = null;
//		try{
//			reader = new BufferedReader(new FileReader(new File(filename)));
//			
//			int inputs, outputs;
//			
//			// The first line of the file will contain the
//			// number of inputs and outputs in each line
//			String line = reader.readLine();
//			Scanner lineScanner = new Scanner(line);
//			inputs = lineScanner.nextInt();
//			outputs = lineScanner.nextInt();
//			
//			ArrayList<String> lineList = new ArrayList<String>();
//			
//			line = reader.readLine();
//			while(line != null) {
//				
//				lineList.add(line);	
//				line = reader.readLine();
//			}
//			
//			double[][] input_array = new double[lineList.size()][inputs];			
//			double[][] output_array = new double[lineList.size()][outputs];
//			
//			for(int i = 0; i < lineList.size(); i++) {
//				// Each line of data will contain inputs first followed
//				// by corresponding training output
//				lineScanner = new Scanner(lineList.get(i).replaceAll(",", " "));
//				
//				for(int j = 0; j < inputs; j++)
//					input_array[i][j] = lineScanner.nextDouble();
//				for(int j = 0; j < outputs; j++)
//					output_array[i][j] = lineScanner.nextDouble();
//				
//				//System.out.println("Data: " + Arrays.toString(input_array[i]));
//						
//			}
//			lineScanner.close();
//			return new TrainingData(input_array, output_array);
//			
//		} catch(FileNotFoundException e) {
//			pl("File error: " + filename + e.toString());
//		} catch(IOException e) {
//			pl("Error processing file: " + e);
//		} finally {
//			try {
//				if(reader != null)
//					reader.close();				
//			} catch(IOException e) {
//				pl("Error closing reader: " + e);
//			}
//		}
//		return null;
//	}
	
	public static MNIST_Data LoadMNISTTrainingData() {
		MNIST_Data training_data = new MNIST_Data();

		training_data.training_input = 
				ReadMNISTImageFile("./data/MNIST/train-images.idx3-ubyte");
		training_data.training_output = 
				ReadMNISTLabelFile("./data/MNIST/train-labels.idx1-ubyte");
		training_data.test_input = 
				ReadMNISTImageFile("./data/MNIST/t10k-images.idx3-ubyte");
		training_data.test_output = 
				ReadMNISTLabelFile("./data/MNIST/t10k-labels.idx1-ubyte");
		
		return training_data;
	}
	
	public static int [] ReadMNISTLabelFile(String filename)
	{
		File file = new File(filename);
		FileInputStream reader = null;
		try{
			reader = new FileInputStream(file);
			
			pl("Reading file: " + filename);
			
			//byte fileContents[] = new byte[(int)file.length()];
			//reader.read(fileContents);
			
			//  http://yann.lecun.com/exdb/mnist/
			//	Training set label file
			//	[offset] [type]          [value]          [description] 
			//	0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
			//	0004     32 bit integer  60000            number of items 
			//	0008     unsigned byte   ??               label 
			//	0009     unsigned byte   ??               label 
			//	........ 
			//	xxxx     unsigned byte   ??               label
			
			// -- Read magic number --
			byte [] magic_number = new byte[4];
			reader.read(magic_number);
			
			// -- Read number of elements --
			byte [] minibuffer = new byte[4];
			reader.read(minibuffer);
			ByteBuffer bb = ByteBuffer.wrap(minibuffer);
			int elements = bb.getInt();
			
			pl("Magic Number = " + magic_number[2] + ", " + magic_number[3]);
			pl("size = " + elements);
			
			// -- Load training data --
			byte [] buffer = new byte[elements];
			int [] training_labels = new int[elements];
			reader.read(buffer);
			for(int i = 0; i < elements; i++) {
				// -- convert into int array
				training_labels[i] = 0xff&buffer[i];
			}
			
			// Test display labels
//			for(int i = 0; i < 100; i++) {
//				pl(training_labels[i]);
//			}
			
			return training_labels;
			
		} catch(FileNotFoundException e) {
			pl("Error loading: " + filename + " - " + e );
		} catch (IOException e) {
			pl("Error reading file: " + e);
		} finally {
			try {
				if(reader != null) 
					reader.close();
			} catch (IOException e) {
				pl("Error closing stream: " + e);
			}
		}
		return null;
	}
	
	public static double[][] ReadMNISTImageFile(String filename)
	{
		File file = new File(filename);
		FileInputStream reader = null;
		try{
			reader = new FileInputStream(file);
			
			pl("Reading file: " + filename);
			
			//byte fileContents[] = new byte[(int)file.length()];
			//reader.read(fileContents);
			
			//  http://yann.lecun.com/exdb/mnist/
			//	Training set image file
			//	[offset] [type]          [value]          [description] 
			//	0000     32 bit integer  0x00000803(2051) magic number 
			//	0004     32 bit integer  60000            number of images 
			//	0008     32 bit integer  28               number of rows 
			//	0012     32 bit integer  28               number of columns 
			//	0016     unsigned byte   ??               pixel 
			//	0017     unsigned byte   ??               pixel 
			//	........ 
			//	xxxx     unsigned byte   ??               pixel
			
			// -- Read magic number --
			byte [] magic_number = new byte[4];
			reader.read(magic_number);
			
			// -- Read number of elements, rows & cols --
			int elements, rows, cols, img_size;
			byte [] minibuffer = new byte[12];
			ByteBuffer bb = ByteBuffer.wrap(minibuffer);
			reader.read(minibuffer);

			elements = bb.getInt();
			rows = bb.getInt();
			cols = bb.getInt();
			img_size = rows * cols;
			
			pl("Magic Number = " + magic_number[2] + ", " + magic_number[3]);
			pl("size = " + elements);
			pl("rows = " + rows);
			pl("cols = " + cols);
			
			
			
			// --- Load training data ---
			double [][] training_images = new double[elements][img_size]; 
			
			byte [] img_buffer = new byte [img_size];
			for(int i = 0; i < elements; i++) {
				reader.read(img_buffer);
				for(int p = 0; p < img_size; p++) {
					// -- convert the unsigned byte to int
					// -- convert value to double where 0 -> 0.0 and 255 -> 1.0
					training_images[i][p] = (double)(0xff&img_buffer[p]) / 256f;
				}	
			}
			
			
			// Test reading
//			for(int i = 0; i < 5; i++) {
//				for(int r = 0; r < rows; r++) {
//					for(int c = 0; c < cols; c++) {
//						p(training_images[i][(r * rows) + c] < 0.1 ? "0" : "1");
//						//p(training)
//					}
//					line();
//				}
//				line();
//			}
			
			return training_images;
			
		} catch(FileNotFoundException e) {
			pl("Error loading: " + filename + " - " + e );
		} catch (IOException e) {
			pl("Error reading file: " + e);
		} finally {
			try {
				if(reader != null) 
					reader.close();
			} catch (IOException e) {
				pl("Error closing stream: " + e);
			}
		}
		return null;
	}
	
// ----------------------------------------
// --------------- Testing ----------------
// ----------------------------------------
	public static void main(String [] args)
	{

		ReadMNISTLabelFile("./data/MNIST/train-labels.idx1-ubyte");
		ReadMNISTImageFile("./data/MNIST/train-images.idx3-ubyte");
		
		
		
	}
    public static void p(Object o){System.out.print(o); }
    public static void pl(Object o){System.out.println(o); }
    public static void line(){System.out.println(); }
}

//	IDX file format: 
//	magic number 
//	size in dimension 0 
//	size in dimension 1 
//	size in dimension 2 
//	..... 
//	size in dimension N 
//	data
