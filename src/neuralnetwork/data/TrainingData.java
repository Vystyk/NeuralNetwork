package neuralnetwork.data;

public abstract class TrainingData {
	
	public int completed_epochs = 0;
	int total_itterations = 0;
	
	public double [][] training_input;
	//public double [][] training_output;
	public int [] training_output;
	

	public double [][] test_input;
	public int [] test_output;
	
	protected int inputs, outputs;
		
	protected int training_index = 0;
	protected int test_index = 0;
	
	public int current_index() { return training_index; }
	
	public int IncremenTrainingIndex() { 
		training_index++;
		if(training_index >= training_input.length) {
			training_index = 0;
			completed_epochs++;
		}
		
		return training_index;
	}
	
	public int IncrementTestIndex() { 
		test_index++;
		if(test_index >= test_input.length) {
			test_index = 0;
		}
		
		return test_index;
	}
	
	public void Reset() {
		completed_epochs = 0;
		total_itterations = 0;
		
		training_index = 0;// Remember to call IncrementIndex
		test_index = 0;
	}
	
	public double [] GetTrainingInput() {
		return training_input[training_index];
	}
	
	public double [] GetTrainingOutput() {
		// convert output answer to categorical double array
		double [] output = new double[outputs];
		output[training_output[training_index]] = 1f;
		return output;
	}
	
	public int GetTrainingOutputIndex() {
		return training_output[training_index];
	}
	
	public double [] getTestInput() {
		return test_input[test_index];
	}
	public int getTestOutput() {
		return test_output[test_index];
	}

}
