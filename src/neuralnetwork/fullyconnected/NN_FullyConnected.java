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

public class NN_FullyConnected extends NeuralNetwork{
	
	public TrainingData training_data;
	
    public int [] structure;
    
    public double training_speed = 1d;
    public double weight_clamp = 100d;
	
	public Layer [] layer;
	int depth;
	int width;
	public Layer output;
	
	
	public NN_FullyConnected(int [] structure){
		this.structure = structure;
		layer = new Layer[structure.length];
		
		InitializeNetwork();
	}
	
	public NN_FullyConnected(int [] structure, double training_speed, double weight_clamp){
		this.structure = structure;
		this.training_speed = training_speed;
		this.weight_clamp = weight_clamp;
		layer = new Layer[structure.length];
		
		InitializeNetwork();
		
	}
	
	public void InitializeNetwork()
	{
		layer[0] = new Layer(structure[0], 0);
		for(int l = 1; l < structure.length; l++){
			layer[l] = new Layer(structure[l], structure[l - 1]);
			layer[l].weight_clamp = weight_clamp;
			layer[l].layer_distance_modifier = Math.sqrt(structure.length - l);
		}
		
		output = layer[structure.length - 1];
		depth = layer.length;
		width = 0;
		for(int l = 0; l < depth; l++)
			if(width < structure[l])
				width = structure[l];

		stats.selected_output = new int[output.value.length][output.value.length];
		stats.percent_correct = new double[output.value.length];
	}
	
	
	public double[] TrainNetwork(double[] input, double[] target_output){
		
		// - Set input neuron values -
		for(int n = 0; n < layer[0].neurons; n++){
			layer[0].value[n] = input[n] > 1.0 ? 1.0 : input[n];
		}
		
		// --- Generate output (feed forward)---
		for(int l = 1; l < layer.length; l++)
		{
			
			for(int n = 0; n < layer[l].neurons; n++){
				
				layer[l].value[n] = 0.0;
				for(int c = 0; c < layer[l].connections; c++){
					layer[l].value[n] += layer[l].weights[n][c] * layer[l - 1].value[c];
				}
				
				layer[l].value[n] = SquashingFunctionSigmoid(layer[l].value[n]);
			}
		}
		
		// --------- Set error for output layer ---------
		// ----- Calculate average error of current trial -----
		stats.training_error = 0;
		for(int n = 0; n < output.neurons; n++){

			double difference = (target_output[n] - output.value[n]);
			double value_factor = output.value[n] * (1.0 - output.value[n]);
			output.error[n] = difference * value_factor;
					
			//System.out.println("error: " + error[n]);
			stats.training_error += Math.abs(output.error[n]);
		}
		stats.error_averager.addElement(stats.training_error);
		
		
		// --------- Backpropagate ---------
		for(int l = layer.length - 1; l > 0; l--)
		{
			// --- Calculate error of previous layer (next in backpropagation) ---
			for(int n = 0; n < layer[l - 1].neurons; n++){
				layer[l - 1].error[n] = 0;
				for(int c = 0; c < layer[l].neurons; c++){
					layer[l - 1].error[n] += layer[l].error[c] * layer[l].weights[c][n];
				}
				layer[l - 1].error[n] *= layer[l - 1].value[n] * (1.0 - layer[l - 1].value[n]);
				//layer[l - 1].error[n] = Math.max(-0.02, Math.min(0.02, layer[l - 1].error[n]));
				
				//System.out.println("error: " + previous_layer.error[n]);
			}
			// --- Adjust each connection weight based upon error ---
			for(int n = 0; n < layer[l].neurons; n++){
				for(int c = 0; c < layer[l].connections; c++){
					
					layer[l].weights[n][c] += training_speed * layer[l].error[n] * layer[l - 1].value[c];
					
					// Clamp
					if(layer[l].weights[n][c] > weight_clamp)
						layer[l].weights[n][c] = weight_clamp;
					else if(layer[l].weights[n][c] < -weight_clamp)
						layer[l].weights[n][c] = -weight_clamp;
				}
			}
			
			
		}
		return output.value;

	}
	
	public double SquashingFunctionSigmoid(double value){
		return 1.0 / ( 1.0 + Math.pow(Math.E, -value));
	}
	
	public double SquashingFunctionSigmoidAprox(double value)
	{	
		if(value < 2.0) {
			if(value > -2.0) {
				return value * 0.2 + 0.5;
			} else if( value > -5.0){
				return (value * (1.0/30.0)) + (1.0/6.0);
			} else
				return 0.0;
		} else {
			if( value < 5.0)
				return (value * (1.0/30.0)) + (5.0/6.0);
			else 
				return 1.0;
		}
	}
	public double CheckError(double[] input, double [] target_output){
		//  --------- Check error, no backpropagaton -----------
		// - Set input neuron values -
		for(int n = 0; n < layer[0].neurons; n++){
			layer[0].value[n] = input[n] > 1f ? 1f : input[n];
		}
		
		// --- Generate output ---
		for(int l = 1; l < layer.length; l++)
		{
			
			for(int n = 0; n < layer[l].neurons; n++){
				
				layer[l].value[n] = 0.0;
				for(int c = 0; c < layer[l].connections; c++){
					layer[l].value[n] += layer[l].weights[n][c] * layer[l - 1].value[c];
				}
				
				layer[l].value[n] = SquashingFunctionSigmoid(layer[l].value[n]);
				//value[n] = SquashingFunctionSigmoidAprox(value[n]);
			}
		}
		
		double total_error = 0;
		for(int n = 0; n < output.neurons; n++)
			total_error += Math.abs(output.error[n]);
		
		return total_error;
	}
	
	public double [] GetOutput(double[] input){
		// Set input neuron values
		for(int n = 0; n < layer[0].neurons; n++){
			layer[0].value[n] = input[n] > 1f ? 1f : input[n];
		}
		
		// --- Generate output (feed forward)---
		for(int l = 1; l < layer.length; l++)
		{
			
			for(int n = 0; n < layer[l].neurons; n++){
				
				layer[l].value[n] = 0.0;
				for(int c = 0; c < layer[l].connections; c++){
					layer[l].value[n] += layer[l].weights[n][c] * layer[l - 1].value[c];
				}
				
				layer[l].value[n] = SquashingFunctionSigmoid(layer[l].value[n]);
				//value[n] = SquashingFunctionSigmoidAprox(value[n]);
			}
		}
		
		return layer[layer.length - 1].value;
	}
	
	public void LoadTrainingData(String filename) {
		//training_data = DataLoader.LoadTrainingData(filename);
		//pl("Loaded: " + filename);
	}
	
	
//	public int TrainWithLoadedData() {
//		int current_index = training_data.IncremenTrainingIndex();
//		
////		// Testing - only train 4 and 9
////		if(training_data.GetTrainingOutputIndex() != 4 && 
////				training_data.GetTrainingOutputIndex() != 9)
////			return current_index;
//		
//		TrainNetwork(
//				training_data.GetTrainingInput(),
//				training_data.GetTrainingOutput());	
//
//		return current_index;
//	}
//	
//	public int TestWithLoadedData() {
//		int current_index = training_data.IncrementTestIndex();
//		
//		
////		// Testing - only test 4 and 9
////		if(training_data.getTestOutput() != 4 && 
////				training_data.getTestOutput() != 9)
////			return current_index;
//		
//		
//		boolean correct = TestNetwork(
//				training_data.getTestInput(),
//				training_data.getTestOutput());	
//		
//		stats.total_tests++;
//		if(correct)
//			stats.correct_tests++;
//
//
//		return current_index;
//	}

	
	public boolean TestNetwork(double[] input, int expected_index)
	{
		
		// - Set input neuron values -
		for(int n = 0; n < layer[0].neurons; n++){
			layer[0].value[n] = input[n] > 1.0 ? 1.0 : input[n];
		}
		
		// --- Generate output (feed forward)---
		for(int l = 1; l < layer.length; l++)
		{
			
			for(int n = 0; n < layer[l].neurons; n++){
				
				layer[l].value[n] = 0.0;
				for(int c = 0; c < layer[l].connections; c++){
					layer[l].value[n] += layer[l].weights[n][c] * layer[l - 1].value[c];
				}
				
				layer[l].value[n] = SquashingFunctionSigmoid(layer[l].value[n]);
				//value[n] = SquashingFunctionSigmoidAprox(value[n]);
			}
		}
		
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
				layer[l - 1].value[n] = SquashingFunctionSigmoid(layer[l - 1].value[n]);
			}
		}
	}
	
	public class Layer {
		
		public int neurons, connections;
		
		public double [] value;
		public double [] bias;
		public double [][] weights;
		// [neuron][connection]
		
		public double [] error;
		//public double [] expected;
		
		public double weight_clamp = 100.0;
		
		public double layer_distance_modifier = 1.0;

		public Layer ( int neurons, int connections){
			this.neurons = neurons;
			this.connections = connections;
			value = new double[neurons];
			bias = new double[neurons];
			
			weights = new double[neurons][connections];
			error = new double[neurons];
			
			for(int n = 0; n < neurons; n++)
				bias[n] = Math.random();
			
			
			Random rng = new Random();
			for(int n = 0; n < neurons; n++)
				for(int c = 0; c < connections; c++)
					weights[n][c] =
						//1f;
						//(Math.random()) * 1.0 - 0.5;
						rng.nextGaussian();
			
		}
	}



	public void SetWeightRegulator(double value) {
		training_speed = value;
		
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
	
	static DecimalFormat df = new DecimalFormat("#.##");
	public static void main(String [] args){

		int itterations = 5000;
//		int [] structure = 
//			{5, 
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
//				5};
		int [] structure = {28*28, 30, 10};
		//int [] structure = {3,3,3};
		NN_FullyConnected nn = new NN_FullyConnected(structure);
		
		//nn.LoadTrainingData("./data/testdata001_5i5o.txt");
		nn.training_data = DataLoader.LoadMNISTTrainingData();
		
		pl("Total neurons: " + nn.CountNeurons());
		pl("Total calculations: " + nn.TotalCalculations());
		
		@SuppressWarnings("unused")
		int output;
		
		for(int i = 0; i < itterations; i++)
		{
			output = nn.TrainWithLoadedData();

			p(nn.training_data.current_index() + " ");
			p(FormatArray(nn.training_data.GetTrainingOutput()) + " ");
			//p(FormatArray(nn.training_data.training_input[nn.training_data.current_index()]));
			//p(FormatArray(nn.training_data.training_output[nn.training_data.current_index()]));
			
			p(FormatArray(nn.currentOutput()));
			p(FormatArray(nn.getOutputErrors()));
			
			
			p(" " + nn.stats.training_error );
			line();
			
			if(i % 100 == 99) {
				pl("    Itterations: " + i);
			}
			
		}
		
		
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
	
    public static void line(){System.out.println(); }
    public static void pl(Object o){System.out.println(o); }
    public static void p(Object o){System.out.print(o); }
    public static int randomInt(int max){return (int)(Math.random() * max);}

	@Override
	public void SetBiasRegulator(double value) {
		// TODO Auto-generated method stub
		
	}






}

