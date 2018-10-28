// ----------------------------------------------------------------------
// ---
// ---	                Victor Jacobson
// ---          2017 CPS Independent Study with Dr. Li
// ---             Neural Network w/ Backpropagation  
// ---  
// ----------------------------------------------------------------------
package neuralnetwork.fullyconnected;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.data.DataLoader;
import neuralnetwork.data.MNIST_Data;
import neuralnetwork.data.Queue;
import neuralnetwork.data.TrainingData;
import neuralnetwork.stats.NetworkStats;

public class NN_FC_ReLU_Bias extends NeuralNetwork{
	
	public TrainingData training_data;
	
    public int [] structure;
    
    public double weight_clamp = 100d;
    
	double regulator = 1.0;
	double step_size = 0.001;
    
	
	public Layer [] layer;
	int depth;
	int width;
	public Layer_Input input_layer;
	public Layer_Output output;
	
	
	public NN_FC_ReLU_Bias(int [] structure){
		this.structure = structure;
		layer = new Layer[structure.length];
		
		InitializeNetwork();
	}
	
	public NN_FC_ReLU_Bias(int [] structure, double regulator, double weight_clamp){
		this.structure = structure;
		this.regulator = regulator;
		this.weight_clamp = weight_clamp;
		layer = new Layer[structure.length];
		
		InitializeNetwork();
		
	}
	
	public void InitializeNetwork()
	{
		layer[0] = new Layer_Input( structure[0]);
		input_layer = (Layer_Input)layer[0];
		
		for(int l = 1; l < structure.length - 1; l++){
			layer[l] = new Layer_Hidden(layer[l - 1], structure[l]);
		}

		layer[ structure.length - 1] = new Layer_Output(layer[structure.length - 2], structure[structure.length - 1]);
		
		output = (Layer_Output) layer[structure.length - 1];
		
		depth = layer.length;
		width = 0;
		for(int l = 0; l < depth; l++)
			if(width < structure[l])
				width = structure[l];

		stats.selected_output = new int[output.value.length][output.value.length];
		stats.percent_correct = new double[output.value.length];
	}
	
	
	public double[] TrainNetwork(double[] input, double[] target_output){
		

		input_layer.SetInput(input);
		
		// --- Generate output (feed forward)---
		for(int l = 1; l < layer.length ; l++)
			layer[l].FeedForward();
		
		// -- check answer --
		int selected = 0;
		double highest = 0.0;
		for(int i = 0; i < output.neurons; i++) {
			if(output.value[i] > highest ) {
				highest = output.value[i];
				selected = i;
			}
		}

		int expected = 0;
		highest = 0.0;
		for(int i = 0; i < output.neurons; i++) {
			if(target_output[i] > highest ) {
				highest = target_output[i];
				expected = i;
			}
		}
		if(expected == selected)
			stats.correct_training++;
		stats.total_training++;
		

		// --- Set error for output layer
		for(int n = 0; n < output.value.length; n++) {
			output.error[n] = output.value[n] - target_output[n];
			
		}
		
		
		// --------- Backpropagate ---------
		for(int l = layer.length - 1; l > 0 ; l--)
			layer[l].Backpropagation();


		return output.value;

	}
	

	public double ReLU(double value){
		return value > 0 ? value : 0.0;
	}
	public double ReLUDer(double value){
		return value > 0 ? 1.0 : 0.0;
	}
	
	
	public double [] GetOutput(double[] input){
		// Set input neuron values
		for(int n = 0; n < layer[0].neurons; n++){
			layer[0].value[n] = input[n] > 1f ? 1f : input[n];
		}
		
		for(int l = 1; l < layer.length - 1; l++)
			layer[l].FeedForward();
		
		return layer[layer.length - 1].value;
	}
	
	public void LoadTrainingData(String filename) {
		//training_data = DataLoader.LoadTrainingData(filename);
		//pl("Loaded: " + filename);
	}
	
	
	
	public boolean TestNetwork(double[] input, int expected_index)
	{
		
		input_layer.SetInput(input);
		
		// --- Generate output (feed forward)---
		for(int l = 1; l < layer.length ; l++)
			layer[l].FeedForward();
		
		
		int answer = 0;
		double highest = 0.0;
		for(int i = 0; i < output.neurons; i++) {
			if(output.value[i] > highest ) {
				highest = output.value[i];
				answer = i;
			}
		}
		//pl(answer);
		
		
		stats.selected_output[expected_index][answer]++;
		
		return answer == expected_index;
	}
	
	public int getTrainingIterations(){ return stats.training_iterations;}

	public double[] getInput() {
		return layer[0].value;
	}

	public Layer [] getLayers() {
		return layer;
	}

	public int getWidth() {
		return width;
	}

	public int getDepth() {
		return depth;
	}
	
	public double[] currentOutput() {
		return layer[layer.length - 1].value;
	}
	
	
	public int CountNeurons() {
		int total = 0;
		for(int l = 0; l < structure.length; l++)
			total += structure[l];
		return total;
	}
	
	public double[] getOutputErrors() {
		return output.error;
	}
	
	public void ReverseOutput(double [] sample_output) {
		// send the output backwards through the network to see the input
		for(int n = 0; n < output.neurons; n++)
			output.value[n] = sample_output[n];
		
		for(int l = layer.length - 1; l > 0; l--) {
			for(int n = 0; n < layer[l-1].neurons; n++) {
				layer[l - 1].value[n] = 0;
				for(int c = 0; c < layer[l].neurons; c++) {
					layer[l - 1].value[n] += layer[l].value[c] * layer[l].weights[c][n];
				}
				layer[l - 1].value[n] = ReLU(layer[l - 1].value[n]);
			}
		}
	}
	
	public abstract class Layer {
		protected Layer input;
		
		public int neurons, connections;
		
		public double [] value;
		public double [] bias;
		public double [][] weights;
		// [neuron][connection]
		
		public double [] error;
		//public double [] expected;
		
		public double weight_clamp = 100.0;
		
		public void InitializeLayer() 
		{
			value = new double[neurons];
			bias = new double[neurons];
			
			weights = new double[neurons][connections];
			error = new double[neurons];
			
			Random rng = new Random();
//			for(int n = 0; n < neurons; n++)
//				bias[n] = rng.nextGaussian();
			
			for(int n = 0; n < neurons; n++)
				for(int c = 0; c < connections; c++)
					weights[n][c] =	rng.nextGaussian() * 0.1;
		}

		
		public abstract void FeedForward();
		
		
		public void Backpropagation()
		{	
			
			// --- Calculate error of previous layer ---
			for(int c = 0; c < connections; c++){
				input.error[c] = 0;
				for(int n = 0; n < neurons; n++){
					input.error[c] += error[n] * weights[n][c];
				}
				
				// ReLU on error based on value
				input.error[c] = input.value[c]  > 0 ? input.error[c] : 0.0;
			}
			
			// --- Adjust each connection weight based upon error ---
			for(int n = 0; n < neurons; n++){
				for(int c = 0; c < connections; c++){

					//double dW = error[n] * input.value[c];
					//dW += regulator * weights[n][c];
					weights[n][c] += -step_size * regulator * error[n] * input.value[c];// * weights[n][c];
					
				}
				// -- Adjust bias --
				bias[n] += -step_size * error[n] * 1.0;
			}
		}
	}
	
	public class Layer_Input extends Layer{
		
		public Layer_Input(int neurons) {
			this.neurons = neurons;
			

			value = new double[neurons];
			bias = new double[neurons];
			
			error = new double[neurons];
		}
		
		public void SetInput(double [] input) {
			this.value = input;
		}

		@Override
		public void FeedForward() {}
		
	}
	
	public class Layer_Hidden extends Layer{

		public Layer_Hidden(Layer input, int neurons) {
			this.input = input;
			this.neurons = neurons;
			connections = input.neurons;
			
			InitializeLayer();

			
		}
		
		public void FeedForward() 
		{
			for(int n = 0; n < neurons; n++){
				
				value[n] = 0.0;
				for(int c = 0; c < connections; c++){
					value[n] += weights[n][c] * input.value[c];
				}
				
				value[n] = ReLU(value[n] + bias[n]);
			}
		}

	}
	
	public class Layer_Output extends Layer{
		
		public Layer_Output(Layer input, int neurons) {
			this.input = input;
			this.neurons = neurons;
			connections = input.neurons;
			value = new double[neurons];
			bias = new double[neurons];
			
			InitializeLayer();
		}
		
		public void FeedForward(){
						
			// ----- Softmax output layer -----
			double exp_sum = 0.0;
			for(int n = 0; n < neurons; n++){
				
				value[n] = 0.0;
				for(int c = 0; c < connections; c++){
					value[n] += weights[n][c] * input.value[c];
				}
				// No ReLU on output layer
				value[n] = Math.exp(value[n] + bias[n]);
				exp_sum += value[n];
			}
			
			for(int n = 0; n < neurons; n++) 
				value[n] /= exp_sum;
			//pl(Arrays.toString(value));
		}

	}


	public void SetWeightRegulator(double value) {
		regulator = value;
		
	}


	public int TotalParameters() {
		int total = 0;
		for(int l = 0; l < structure.length - 1; l++)
			total += structure[l] * structure[l + 1];
		return total;
	}

	
	public int TotalCalculations() {
		
		int total = 0;
		for(int l = 0; l < structure.length - 1; l++)
			total += structure[l] * structure[l + 1] + structure[l];
		
		total += structure[structure.length - 1];// add all the neurons
		
		return total;
	
	}

	public String Structure() {

		return Arrays.toString(structure);
	}

		@Override
	public boolean TestNetwork(double[] input, double[] target_output) {
		// TODO Auto-generated method stub
		return false;
	}
		
	@Override
	public void TrainNetwork(double[] input, int expected_index) {
		// TODO Auto-generated method stub
		
	}
	
	

	
// -----------------------------------------------------------------------------------
// ---                                   Testing                                   ---
// -----------------------------------------------------------------------------------
//	
	static DecimalFormat df = new DecimalFormat("#.##");
//	public static void main(String [] args){
//
//		int itterations = 5000;
//		int [] structure = {28*28, 30, 10};
//		//int [] structure = {3,3,3};
//		NN_FC_ReLU_Bias nn = new NN_FC_ReLU_Bias(structure);
//		
//		//nn.LoadTrainingData("./data/testdata001_5i5o.txt");
//		nn.training_data = DataLoader.LoadMNISTTrainingData();
//		
//		pl("Total neurons: " + nn.CountNeurons());
//		pl("Total calculations: " + nn.TotalCalculations());
//		
//		int output;
//		
//		for(int i = 0; i < itterations; i++)
//		{
//			output = nn.TrainWithLoadedData();
//
//			p(nn.training_data.current_index() + " ");
//			p(FormatArray(nn.training_data.GetTrainingOutput()) + " ");
//			//p(FormatArray(nn.training_data.training_input[nn.training_data.current_index()]));
//			//p(FormatArray(nn.training_data.training_output[nn.training_data.current_index()]));
//			
//			p(FormatArray(nn.currentOutput()));
//			p(FormatArray(nn.getOutputErrors()));
//			
//			
//			p(" " + nn.stats.training_error );
//			line();
//			
//			if(i % 100 == 99) {
//				pl("    Itterations: " + i);
//			}
//			
//		}
//		
//		
//	}
	
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
	
    public static void line(){System.out.println(); }
    public static void pl(Object o){System.out.println(o); }
    public static void p(Object o){System.out.print(o); }
    public static int randomInt(int max){return (int)(Math.random() * max);}

	@Override
	public void SetBiasRegulator(double value) {
		// TODO Auto-generated method stub
		
	}

	

}

