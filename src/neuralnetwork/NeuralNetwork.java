package neuralnetwork;


import neuralnetwork.data.TrainingData;
import neuralnetwork.stats.NetworkStats;

public abstract class NeuralNetwork {

	public static enum MODE{TRAINING, TESTING, COMPLETE};
	
	public MODE mode;
	
	protected int neuron_count;
	protected int parameter_count;
	protected int calculations;
	
	TrainingData training_data;
	public double [] output;
	
	protected NetworkStats stats = new NetworkStats();
	
	public abstract double[] TrainNetwork(double [] input, double [] target_output);
	public abstract void TrainNetwork(double [] input, int target_index);
	
	public abstract boolean TestNetwork(double [] input, double [] target_output);
	public abstract boolean TestNetwork(double [] input, int target_index);

	public abstract double [] GetOutput(double [] input);

	public abstract void ReverseOutput(double [] output);
	


	
	public int TrainWithLoadedData() {
		mode = MODE.TRAINING;
		int current_index = training_data.IncremenTrainingIndex();
//		// Testing - only train 4 and 9
//		if(training_data.GetTrainingOutputIndex() != 4 && 
//				training_data.GetTrainingOutputIndex() != 9)
//			return current_index;
		
		output = TrainNetwork(
				training_data.GetTrainingInput(),
				training_data.GetTrainingOutput());	
		
		stats.training_iterations++;
		
		
		
		return current_index;
	}
	
	public int TestWithLoadedData() {
		mode = MODE.TESTING;
		int current_index = training_data.IncrementTestIndex();
		

//		// Testing - only test 4 and 9
//		if(training_data.getTestOutput() != 4 && 
//				training_data.getTestOutput() != 9)
//			return current_index;
		
		
		boolean correct = TestNetwork(
				training_data.getTestInput(),
				training_data.getTestOutput());	
		
		stats.total_tests++;
		if(correct)
			stats.correct_tests++;


		return current_index;
	}

	public void SetTrainingData(TrainingData td) {
		training_data = td;
		
	}

	public TrainingData GetTrainingData() {
		return training_data;
	}

	public NetworkStats GetStats() {
		return stats;
	}
	
	

	public abstract String Structure();
	public int TotalParameters() {return parameter_count;}
	public int TotalCalculations() {return calculations;}


	public abstract void SetWeightRegulator(double value);
	public abstract void SetBiasRegulator(double value);
	
	
}
