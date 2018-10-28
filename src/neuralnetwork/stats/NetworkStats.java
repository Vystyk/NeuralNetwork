package neuralnetwork.stats;

import neuralnetwork.data.Queue;

public class NetworkStats{
	public int training_iterations = 0;
	public Queue<Double> error_averager = new Queue<Double>();// queue error of previous training inputs.

	
	public double training_error;
	//public int selected_output;
	
	
	public int total_tests = 0;
	public int correct_tests = 0;

	public int total_training = 0;
	public int correct_training = 0;
	
	public long elapsed_time = 0;
	
	
	public int [][] selected_output; // selected per intended output
	public double [] percent_correct;
	
	
	public double AverageError() {

		return error_averager.average();
	}
	
	// Call after a complete run of the test data
	public void CalPercentSelectedOutput() {
		int total = 0;
		for(int i = 0; i < selected_output.length; i++) {
			for(int j = 0; j < selected_output.length; j++) {
				total += selected_output[i][j];
			}
			percent_correct[i] = ((int)((float)selected_output[i][i] / (float)total * 10000)) / 100.0;
			total = 0;
		}
	}
	
	public void ResetTestResults() {
		int length = selected_output.length;
		selected_output = new int[length][length];
		
		correct_tests = 0;
		total_tests = 0;

		total_training = 0;
		correct_training = 0;
		
	}

}