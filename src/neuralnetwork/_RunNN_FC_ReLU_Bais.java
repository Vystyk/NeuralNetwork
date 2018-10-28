package neuralnetwork;


import java.awt.event.KeyEvent;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Observable;

import javafx.animation.AnimationTimer;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Duration;
import neuralnetwork.data.DataLoader;
import neuralnetwork.data.MNIST_Data;
import neuralnetwork.fullyconnected.NN_FC_ReLU_Bias;
import neuralnetwork.fullyconnected.NN_FullyConnected;
import neuralnetwork.logs.LogFileWriter;
import neuralnetwork.render.NN_RenderJFX;
import javafx.scene.control.*;
import javafx.util.Callback;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.event.*;


public class _RunNN_FC_ReLU_Bais extends Application{
	
	float WIDTH = 1800;
	float HEIGHT = 1000;
	
	float nn_x = 100;
	float nn_y = 100;
	float nn_width = 1600;
	float nn_height = 800;
	
	NeuralNetwork nn;
	
	MNIST_Data mnist_data;
	
	GraphicsContext gc;
	
	long previousNanoTime;
	long frame_time = 0;

	Text itterations_text = new Text( 1000, 35, "");
	Text error_text = new Text( 1000, 50, "Error");
	Text error2_text = new Text( 1000, 65, "Error current dataset:");

	Text epoch_text1 = new Text( 800, 35, "");
	Text epoch_text2 = new Text( 800, 50, "");
	Text epoch_text3 = new Text( 800, 65, "");
	Text epoch_text4 = new Text( 800, 80, "");

	Button pause_button = new Button("Run");
	Button speed_button = new Button("Fast");
	Button step_button = new Button("Step");
	Button reverse_button = new Button("Reverse");
	//Button repeat_button = new Button("0");
	//Button learning_rate_button = new Button("1.0");
	
	boolean running = false;
	boolean full_speed = true;
	boolean step = false;
	

	double [] reverse_input_output = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	
	public static void main(String [] args){
		launch(args);
	}
	
	@SuppressWarnings("restriction")
	@Override
	public void start(Stage primaryStage) {
		primaryStage.setTitle("Neural Network");
		Group root = new Group();
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		
		canvas.addEventHandler(MouseEvent.MOUSE_CLICKED, new EventHandler<MouseEvent>() {
			@Override
			public void handle(final MouseEvent e){
				if(e.getButton() == MouseButton.SECONDARY){
					
					
				} else if(e.getButton() == MouseButton.MIDDLE) {
					
					
				} else if(e.getButton() == MouseButton.PRIMARY){
					
					int n = getInputNeuron(e.getX(), e.getY());
					if(n >= 0 && n < 5){
						//nn.layer[0].neurons[n].value = toggle(nn.layer[0].neurons[n].value);
						//nn.ProcessNetwork();
					} 
				}	
				
				DrawNeuralNetwork();
				
			}
		});
		
		gc = canvas.getGraphicsContext2D();
		NN_RenderJFX.SetGC(gc);
		
		

				
		root.getChildren().add(canvas);

		root.getChildren().add(error_text);
		root.getChildren().add(error2_text);
		root.getChildren().add(itterations_text);
		
		root.getChildren().add(epoch_text1);
		root.getChildren().add(epoch_text2);
		root.getChildren().add(epoch_text3);
		root.getChildren().add(epoch_text4);
		
//		BorderPane border = new BorderPane();
//		border.setPadding(new Insets(20,0,20,20));
		VBox buttons = new VBox();
		buttons.setSpacing(10);
		//buttons.set
		
		root.getChildren().add(new Text(500, 20, "                  Victor Jacobson 2017 \nNeural Network with Backpropagation"));
		
		root.getChildren().add(buttons);

		// --------------------- Buttons ------------------------
		buttons.getChildren().add(new Text("Network Structure"));
		ComboBox<String> structure_comboBox = new ComboBox<String>();
		buttons.getChildren().add(structure_comboBox);
		HashMap<String, int[]> structureMap = new HashMap<String, int[]>();
		for(int i = 0; i < structures.length; i++) {
			structure_comboBox.getItems().add(Arrays.toString(structures[i]));
			structureMap.put(Arrays.toString(structures[i]), structures[i]);
		}
		structure_comboBox.setValue(Arrays.toString(structures[0]));
		structure_comboBox.valueProperty().addListener(new ChangeListener<String>() {
			@Override
			public void changed(ObservableValue ov, String s1, String s2) {
				//nn = new NeuralNetwork(structureMap.get((String)ov.getValue()));
				//nn.training_data = mnist_data;

				InitializeNeuralNetwork(structureMap.get((String)ov.getValue()));
			}
		});
		
		buttons.getChildren().add(pause_button);
		pause_button.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event){
				running = !running;
				if(running)
					pause_button.setText("Pause");
				else
					pause_button.setText("Run");
			}
		});
		buttons.getChildren().add(speed_button);
		speed_button.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event){
				full_speed = !full_speed;
				if(full_speed)
					speed_button.setText("Fast");
				else
					speed_button.setText("Slow");
			}
		});
		buttons.getChildren().add(step_button);
		step_button.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event){
				step = true;
			}
		});
		buttons.getChildren().add(reverse_button);
		reverse_button.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event){
			
				nn.ReverseOutput(reverse_input_output);
			}
		});


		buttons.getChildren().add(new Text("Reverse Input"));
		ComboBox<Integer> reverseInput_comboBox = new ComboBox<Integer>();
		buttons.getChildren().add(reverseInput_comboBox);
		reverseInput_comboBox.getItems().addAll(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
		reverseInput_comboBox.setValue(0);
		reverseInput_comboBox.valueProperty().addListener(new ChangeListener<Integer>() {
			@Override
			public void changed(ObservableValue ov, Integer i1, Integer i2) {
				int selected = (int)ov.getValue();

				reverse_input_output = new double[10];
				reverse_input_output[selected] = 1.0;
				nn.ReverseOutput(reverse_input_output);
			}
		});
		

		buttons.getChildren().add(new Text("Training Speed"));
		ComboBox<Float> trainingSpeed_comboBox = new ComboBox<Float>();
		buttons.getChildren().add(trainingSpeed_comboBox);
		trainingSpeed_comboBox.getItems().addAll(1.0f, 1.5f, 2f, 5f, 10f, 100f, 0.5f, 0.1f);
		trainingSpeed_comboBox.setValue(1.0f);
		trainingSpeed_comboBox.valueProperty().addListener(new ChangeListener<Float>() {
			@Override
			public void changed(ObservableValue ov, Float f1, Float f2) {
				nn.SetWeightRegulator((float)ov.getValue());
			}
		});
		
		
		primaryStage.setScene(new Scene(root));
		
		LoadTrainingData();
		InitializeNeuralNetwork(structures[0]);
		
		// ----------------- Main Loop -----------------
		previousNanoTime = System.nanoTime();
		frame_time = 0;
		
		new AnimationTimer()
		{
			public void handle(long currentNanoTime)
			{
				ProcessNetwork();
				long elapsed = currentNanoTime - previousNanoTime;
				previousNanoTime = currentNanoTime;
				frame_time += elapsed;
				if(frame_time > 33_333_333) {
					frame_time = 0;
					//if(update_rendering)
						DrawNeuralNetwork();
				}
				
			}
		}.start();
		
		
		
		
		primaryStage.show();
		
		
		primaryStage.setOnCloseRequest(new EventHandler<WindowEvent>() {
			public void handle(WindowEvent we) {
				pl("Closing file.");
				
				LogFileWriter.write("Total itterations: " + nn.GetStats().training_iterations);
				LogFileWriter.close();
			}
		});  
		
	}
	
	private void ProcessNetwork(){
		
		if(running){
			if(full_speed){
				TrainingUpdateFullSpeed();
			} else {
				NetworkUpdate();
			}
		}else{
			if(step){
				NetworkUpdate();
				step = false;
			}
		}
		
	}
	
	
	private void TrainingUpdateFullSpeed() {
		
		long frame_time = 1666666;
		long start_time = System.nanoTime();
		long elapsed_time = 0;
		boolean exit_loop = false;
		
		while(!exit_loop)
		{
			
			NetworkUpdate();
			
			elapsed_time += System.nanoTime() - start_time;
			start_time = System.nanoTime();
			if(elapsed_time > frame_time)
				exit_loop = true;
		}
	}
	
	boolean test_mode = false;
	
	private void NetworkUpdate() {
		long start_time = System.nanoTime(); 
		if(!test_mode) {
			
			if(nn.TrainWithLoadedData() == 0) {
				test_mode = true;
				
				
				LogFileWriter.write("Epoch " + nn.GetTrainingData().completed_epochs + " complete, time: " + 
						LogFileWriter.FormatNanoSeconds(nn.GetStats().elapsed_time));
				double percent = Math.round((double)nn.GetStats().correct_training/nn.GetStats().total_training * 10000.0) / 100.0;
				LogFileWriter.write("    Training: " + nn.GetStats().correct_training + "/" + nn.GetStats().total_training + " - " + percent + "%" );
			}
		} else {
			
			if(nn.TestWithLoadedData() == 0) {
				test_mode = false;
				LogFileWriter.write("Test after " + nn.GetTrainingData().completed_epochs + " epochs, time: " +
						LogFileWriter.FormatNanoSeconds(nn.GetStats().elapsed_time));
				double percent = Math.round((double)nn.GetStats().correct_tests/nn.GetStats().total_tests * 10000.0) / 100.0;
				LogFileWriter.write("    Test: " + nn.GetStats().correct_tests +"/"+ nn.GetStats().total_tests + " - " + percent + "%" );
				int [][] test_results = nn.GetStats().selected_output;
				nn.GetStats().CalPercentSelectedOutput();
				double [] percents = nn.GetStats().percent_correct;
				int size = test_results.length;
				for(int i = 0; i < size; i++)
					LogFileWriter.write(" " + i + " " + Arrays.toString(test_results[i]) + " " + percents[i]);
				
				
				percent = Math.round((double)nn.GetStats().correct_training/nn.GetStats().total_training * 10000.0) / 100.0;
				epoch_text3.setText("Prev Train: " + nn.GetStats().correct_training + "/" + nn.GetStats().total_training + " - " + percent + "%" );
				percent =  Math.round((double)nn.GetStats().correct_tests/nn.GetStats().total_tests * 10000.0) / 100.0;
				epoch_text4.setText("Prev Test: " + nn.GetStats().correct_tests + "/" + nn.GetStats().total_tests + " - " + percent + "%" );
				nn.GetStats().ResetTestResults();
				
			}
			
			
		}

		itterations_text.setText("" + nn.GetStats().training_iterations);
		error_text.setText("Error: " + nn.GetStats().AverageError());
		error2_text.setText("Error: " + nn.GetStats().training_error);
		
		double percent = Math.round((double)nn.GetStats().correct_training/nn.GetStats().total_training * 10000.0) / 100.0;
		epoch_text1.setText("Training: " + nn.GetStats().correct_training + "/" + nn.GetStats().total_training + " - " + percent + "%" );
		percent =  Math.round((double)nn.GetStats().correct_tests/nn.GetStats().total_tests * 10000.0) / 100.0;
		epoch_text2.setText("Test: " + nn.GetStats().correct_tests + "/" + nn.GetStats().total_tests + " - " + percent  + "%" );

		
		
		nn.GetStats().elapsed_time += System.nanoTime() - start_time; 
	}
	
	private int getInputNeuron(double x, double y) {
		
		
		int num = (int) ((y - 40) / 150);

		return num;
	}
	
	public float toggle(float value){
		if(value > 0.5)
			return -1;
		else if(value < -0.5)
			return 0;
		else return 1;
	}
	
	
	
//	int input_set = 0;
	int repeat = 0;
	int repeat_times = 0;
	
	int [][] structures = {
			{ 784, 30, 10},
			{ 784, 50, 10},
			{ 784, 100, 10},
			{ 784, 16, 16, 10},
			{ 784, 30, 30, 10},
			{ 784, 30, 30, 30, 10},
			{ 784, 30, 30, 30, 30, 10},
			{ 784, 40, 30, 30, 20, 10},
			{ 784, 800, 400, 150, 50, 10},
			{ 784, 2500, 2000, 1500, 1000, 500, 10},
		};
	
	private void InitializeNeuralNetwork(int [] structure) {
		
		DateFormat df = new SimpleDateFormat("yyyyMMdd_hhmmss");
		Date date = new Date();
		LogFileWriter.CreateLogFile("mnist_tlog" + df.format(date) + ".txt");
		df = new SimpleDateFormat("yyyy.MM.dd hh:mm:ss");
		LogFileWriter.write(df.format(date));

		nn = new NN_FC_ReLU_Bias(structure);

		LogFileWriter.write("Network Structure: " + nn.Structure());
		LogFileWriter.write("Network Calcultions: " + nn.TotalCalculations());
		LogFileWriter.write("Network Parameters: " + nn.TotalParameters());
		
		mnist_data.Reset();
		nn.SetTrainingData(mnist_data);

		DrawNeuralNetwork();
		
	}
	
	private void LoadTrainingData() {
		
		mnist_data = DataLoader.LoadMNISTTrainingData();
		
	}
	
	private void DrawNeuralNetwork() {
		gc.clearRect(0, 0, WIDTH, HEIGHT);
		
		NN_RenderJFX.RenderNN_FC_ReLU_Bias((NN_FC_ReLU_Bias)nn, nn_x, nn_y, nn_width, nn_height);
	}
	


	

    public static void pl(Object o){System.out.println(o); }

}
