package neuralnetwork.render;

import java.util.Arrays;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import neuralnetwork.NeuralNetwork.MODE;
import neuralnetwork.fullyconnected.NN_FC_ReLU_Bias;
import neuralnetwork.fullyconnected.NN_FullyConnected;

public class NN_RenderJFX {
	
	private static GraphicsContext gc;
	
	public static void SetGC(GraphicsContext gc) {
		NN_RenderJFX.gc = gc;
		Bars.SetGC(gc);
	}
	
	public static void RenderNN_FullyConnected(NN_FullyConnected nn, float startx, float starty, float width, float height) {
		
		int layers = nn.layer.length;
		float endx = startx + width;
		float endy = starty + height;
		
		float incx = (endx - startx) / (layers - 1);
		float incy;// calc for each layer
		

		// Draw hidden layer synapses
		for(int l = 2; l < layers; l++){
			int size = nn.layer[l].neurons;
			incy = (endy - starty) / (size - 1);
			for(int n = 0; n < size; n++){
				for(int c = 0; c < nn.layer[l].connections; c++){
					float incsy = (endy - starty) / (nn.layer[l].connections - 1);
					DrawSynapse(startx + (incx * l), starty + (incy * n),
							startx + (incx * (l - 1)), starty + (incsy * c),
							//truncate((int)(nn.layer[l].weights[n][c] * 256f) + 127, 0, 255),
							nn.layer[l].weights[n][c],
							Math.abs(nn.layer[l].weights[n][c] * 2f),
							nn.layer[l-1].value[c] * nn.layer[l].value[n]);
				}
			}
		}
		
		// Draw Input synapses
				{
				int l = 1;
					int size = nn.layer[1].neurons;
					incy = (endy - starty) / (size - 1);
					for(int n = 0; n < size; n++){
						
						for(int r = 0; r < 28; r++) {
							float y = starty + r * 10;
							for(int c = 0; c < 28; c++) {
								int con = 28 * r + c;
								float x = startx + c * 10;
								//float incsy = (endy - starty) / (nn.layer[l].connections - 1);
								DrawSynapse(startx + (incx * l), starty + (incy * n),
									x, y,
									//truncate((int)(nn.layer[l].weights[n][c] * 256f) + 127, 0, 255),
									nn.layer[l].weights[n][c],
									(float) Math.abs(nn.layer[l].weights[n][c] * 2f),
									(float)(nn.layer[0].value[con] * nn.layer[1].value[n]));
							}
						}
					}
				}
		// Draw Hidden Neurons
		for(int l = 1; l < layers; l++){
			int size = nn.layer[l].neurons;
			incy = (endy - starty) / (size - 1);
			for(int n = 0; n < size; n++){
				float value = (float) nn.layer[l].value[n];
				DrawNeuron(startx + (incx * l), starty + (incy * n), 
						(float)Math.sqrt(Math.abs(value * 5f))* 3f + 20f,
						value,
						 (float) nn.layer[l].error[n]);
			}
		}
		

		
		// Draw Input Neurons
		for(int r = 0; r < 28; r++) {
			float y = starty + r * 10;
			for(int c = 0; c < 28; c++) {
				float x = startx + c * 10;
				int n = 28 * r + c;
				float value = (float) nn.layer[0].value[n];
				DrawNeuron(x, y, 
						(float)Math.sqrt(Math.abs(value * 5f))* 3f + 10f,
						value,
						 (float) nn.layer[0].error[n]);
			}
		}
	}
	
	
	private static void DrawOutputLayer(float x, float y, float height, double [] value, double [] error, int correct) {
		 // Draw Neurons
		float start_x = x;
		float y_inc = (height / value.length);
		
		int selected = 0;
		double max = 0;
		for(int n = 0; n < value.length ; n++) {
			if(n == correct) {
				// Highlight correct output
				gc.setStroke(Color.DARKORANGE);
				gc.setLineWidth(3f);
				gc.strokeRect(start_x - 15, y + y_inc * n - 15, 30, 30);
			}
			DrawNeuron(start_x , y + y_inc * n, 
				(float)Math.sqrt(Math.abs(value[n] * 5f))* 3f + 20f,
				(float)value[n],
				(float)error[n]);
			
			if(value[n] > max) {
				max = value[n];
				selected = n;
			}
							
		}
		// Highlight selected output
		gc.setStroke(Color.DARKGOLDENROD);
		gc.setLineWidth(7f);
		gc.strokeOval(start_x - 10, y + y_inc * selected - 10, 20, 20);
		gc.setStroke(Color.YELLOW);
		gc.setLineWidth(5f);
		gc.strokeOval(start_x - 10, y + y_inc * selected - 10, 20, 20);
		
	}
	
	@SuppressWarnings("unused")
	public static void RenderNN_FC_ReLU_Bias(NN_FC_ReLU_Bias nn, float startx, float starty, float width, float height) {
		
		int layers = nn.layer.length;
		double endy = starty + height;
		
		double inc_layer_x = (width * 0.9) / (layers - 1);
		double incy1;// calc for each layer
		
		// Draw Input synapses	
		int l = 1;
		int size = nn.layer[1].neurons;
		incy1 = (endy - starty) / (size);
		if(true) {
		for(int n = 0; n < size; n++){

			for(int r = 0; r < 28; r++) {
				float y = starty + r * 10;
				for(int c = 0; c < 28; c++) {
					int con = 28 * r + c;
					double x = startx + c * 10;
					//double incsy = (endy - starty) / (nn.layer[l].connections);
					DrawSynapse(startx + (inc_layer_x * l), starty + (incy1 * n) + incy1 * 0.5,
							x, y,
							//truncate((int)(nn.layer[l].weights[n][c] * 256f * 10.0) + 128, 0, 255),
							nn.layer[l].weights[n][c],
							Math.abs(nn.layer[l].weights[n][c] * 2.0 + 0.2),
							nn.layer[0].value[con] * nn.layer[1].value[n]);
				}
			}
		}
		
		// Draw inner synapses
		for(l = 2; l < layers; l++){
			size = nn.layer[l].neurons;
			incy1 = (endy - starty) / (size);
			for(int n = 0; n < size; n++){
				for(int c = 0; c < nn.layer[l].connections; c++){
					
					double incy2 = (endy - starty) / (nn.layer[l].connections);
					DrawSynapse(startx + (inc_layer_x * l), starty + (incy1 * n) + incy1 * 0.5,
							startx + (inc_layer_x * (l - 1)) + (endy - starty) / nn.layer[l - 1].neurons, starty + (incy2 * c) + incy2 * 0.5,
							//truncate((int)(nn.layer[l].weights[n][c] * 256f) + 128, 0, 255),
							nn.layer[l].weights[n][c],
							Math.abs(nn.layer[l].weights[n][c] * 2.0 + 0.2),
							nn.layer[l-1].value[c] * nn.layer[l].value[n]);
				}
			}
		}
		}
				
		// Draw Input Neurons
		//TODO use a 2d or 3d layer format
		for(int r = 0; r < 28; r++) {
			float y = starty + r * 10;
			for(int c = 0; c < 28; c++) {
				int n = 28 * r + c;
				DrawNeuron(startx + c * 10, y, 
						10.0,
						nn.layer[0].value[n],
						nn.layer[0].error[n] * 0.001
						);
			}
		}
		
		// ---- Draw inner neurons ----
		for(l = 1; l < layers - 1; l++){
			
			DrawHiddenLayerBars(startx + (inc_layer_x * l), starty, 
						(endy - starty) / nn.layer[l].neurons, height,
						nn.layer[l].value,
						nn.layer[l].bias,
						nn.layer[l].error);
		}
		
		
		// ---- Output Layer ----
		int correct_index = 
				nn.mode == MODE.TRAINING ? 
				nn.GetTrainingData().GetTrainingOutputIndex() : nn.GetTrainingData().getTestOutput() ;
				
		DrawOutputLayerSoftmax(
				//startx + width * output_layer_size_factor, 
				startx + inc_layer_x * (layers - 1),
				starty, width / 15, height, 
				nn.output.value,
				nn.output.bias,
				nn.output.error,
				correct_index);


	}

	private static void DrawHiddenLayerBars(double x, double y, double w, double h, double [] value, double [] bias, double [] error) {
		
		double y_inc = (h / value.length);
		y += y_inc * 0.1;
		if(y_inc > 2)
		{
			for(int n = 0; n < value.length ; n++) {
				
				DrawNeuronBar(x , y + (y_inc * n), 
					w, y_inc * 0.8,
					value[n],
					bias[n],
					error[n]);
								
			}
		} else {
			for(int n = 0; n < value.length ; n++) {
				gc.setFill(Color.BLACK.interpolate(Color.LAWNGREEN, value[n] * 0.5));
				gc.fillRect(x, y + y_inc * n, w, y_inc);
			}
		}
	}
	
	
	protected static void DrawOutputLayerSoftmax(double x, double y, double w, double h, double [] value, double [] bias, double [] error, int correct) {
		
		double y_inc = (h / value.length);
		y += y_inc * 0.1;
		
		int selected = 0;
		double max = 0;
		for(int n = 0; n < value.length ; n++) {
			
			DrawNeuronBar(x , y + (y_inc * n), 
				w, y_inc * 0.8,
				value[n],
				bias[n],
				error[n]);
			
			if(value[n] > max) {
				max = value[n];
				selected = n;
			}
							
		}
		// Highlight selected output
		gc.setStroke(Color.DARKORANGE);
		gc.setLineWidth(4.0);
		gc.strokeRect(x - 3.0, y + y_inc * selected - 3.0, w + 6.0, y_inc * 0.8 + 6.0);
		gc.setFill(Color.DARKORANGE);
		gc.fillText("" + selected, x + w * 0.5, y + y_inc * selected + y_inc * 0.5);
		
		// Highlight correct output
		gc.setStroke(Color.DARKGOLDENROD);
		gc.setLineWidth(7f);
		gc.strokeRect(x, y + y_inc * correct, w, y_inc * 0.8);
		gc.setStroke(Color.YELLOW);
		gc.setLineWidth(5f);
		gc.strokeRect(x, y + y_inc * correct, w, y_inc * 0.8);
		gc.setFill(Color.YELLOW);
		gc.fillText("" + correct, x + w * 0.5, y + y_inc * correct + y_inc * 0.5);
		
	}
	
	
	
	private static void DrawNeuronBarOld(double x, double y, double w, double h, double value, double bias, double error){
		// -- Background --
		gc.setFill(Color.BLACK);
		gc.fillRect(x, y, w, h);
		
		// -- Value bar --	
		gc.setFill(Color.FORESTGREEN.interpolate(Color.LAWNGREEN, value * 0.1));
		gc.fillRect( x + w * 0.2, y, w * Math.min(value, 1.0) * 0.6, h );
		

		// -- Bias bar --
		if(bias > 0) {
			bias = Math.min(bias, 1.0);
			gc.setFill(Color.BLACK.interpolate(Color.LIGHTBLUE, bias * 5.0));
		} else if(bias < 0){
			bias = Math.max(bias, -1.0);
			gc.setFill(Color.BLACK.interpolate(Color.DARKVIOLET, -bias * 5.0));
		}
		gc.fillRect( x, y, w * 0.2, h);
		
//		// -- Error bar			
		if(error > 0) {
		error = Math.min(error, 1.0);
		gc.setFill(Color.BLACK.interpolate(Color.RED, error + 0.3));
	} else if(error < 0){
		error = Math.min(-error, 1.0);
		gc.setFill(Color.BLACK.interpolate(Color.BLUE, error + 0.3));
	}
	gc.fillRect( x + w * 0.8, y ,w * 0.2, h);
//		if(error > 0) {
//		error = Math.min(error, 1.0);
//		gc.setFill(Color.RED);
//		gc.fillRect( 
//				x + w * 0.8, y + h * error * 0.5,
//				w * 0.2, h * error * 0.5);
//	} else {
//		error = Math.max(error, -1.0);
//		gc.setFill(Color.BLUE);
//		gc.fillRect( 
//				x + w * 0.8, y + h * 0.5,
//				w * 0.2, h * -error * 0.5);
//	}

	}
	
	private static void DrawNeuronBar(double x, double y, double w, double h, double value, double bias, double error){
		
		double rounded = w * 0.4;
		gc.setFill(Color.BLACK);

		// -- Bias bar --
		if(bias > 0) {
			bias = Math.min(bias, 1.0);
			gc.setFill(Color.BLACK.interpolate(Color.LIGHTBLUE, bias * 5.0));
		} else if(bias < 0){
			bias = Math.max(bias, -1.0);
			gc.setFill(Color.BLACK.interpolate(Color.DARKVIOLET, -bias * 5.0));
		}
		gc.fillRoundRect( x, y, w * 0.6, h, rounded, rounded);
		
		// -- Error bar			
			if(error > 0) {
			error = Math.min(error, 1.0);
			gc.setFill(Color.BLACK.interpolate(Color.RED, error + 0.3));
		} else if(error < 0){
			error = Math.min(-error, 1.0);
			gc.setFill(Color.BLACK.interpolate(Color.BLUE, error + 0.3));
		}
		gc.fillRoundRect( x + w * 0.4, y ,w * 0.6, h, rounded, rounded);
		
//		// -- Value bar --
//		if(value < 1.0)
//			gc.setFill(Color.BLACK.interpolate(Color.FORESTGREEN, value * 1.0));
//		else
//			gc.setFill(Color.FORESTGREEN.interpolate(Color.LAWNGREEN, (value - 1.0) * 0.1));
//		gc.fillRoundRect( x + w * 0.2, y, w * 0.6, h, rounded * 0.5, rounded * 0.5);
		
		// -- Value bar --	
		// -- Background --
		gc.setFill(Color.BLACK);
		gc.fillRect( x + w * 0.2, y, w * 0.6, h);
		gc.setFill(Color.FORESTGREEN.interpolate(Color.LAWNGREEN, value * 0.1));
		gc.fillRect( x + w * 0.2, y, w * Math.min(value, 1.0) * 0.6, h );

	}

	
	
	protected static void DrawNeuron(double x, double y, double d, double value, double error){
		value = Math.max(Math.min(1f, value), 0.0f);
		gc.setFill(Color.rgb(0, (int)(value * 255f), (int)(200 - (value * 0.6))));
		
		gc.fillOval( x - d * 0.5, y - d * 0.5, d, d);
		
		// Draw error bar
		error *= 50f;
		gc.setFill(Color.rgb(255, 0,0));
		
		gc.fillRect(
				x - d / 6f, 
				y + (error > 0.0 ? - error * 2f :  0.0),  
				d / 3f, 
				Math.abs(error) * 2f);
		//System.out.println(error);
		//pl("error: " + error);
	}
	
	private static void DrawNeuronBias(double x, double y, double w, float value, double error, double bias){
		value = Math.max(Math.min(1f, value), 0.0f);
		error = Math.max(Math.min(1f, error), -1.0f);
		w = 15f;
		gc.setFill(Color.rgb(0, (int)(value * 255f), (int)(200 - (value * 0.6))));
		
		gc.fillOval( x - w/2f, y - w/2f, w, w);
		
		// Draw bias ring
		gc.setStroke(Color.rgb(120, 255, 120));
		gc.setLineWidth(3f + (bias * 2f));
		gc.strokeOval( x - w/2f, y - w/2f, w, w);
		
		// Draw error bar
		error *= 5f;
		gc.setFill(Color.rgb(255, 0,0));
		gc.fillRect(
				x - w / 6f, 
				y + (error > 0.0 ? - error * 2f :  0.0),  
				w / 3f, 
				Math.abs(error) * 2f);
		//System.out.println(error);
		//pl("error: " + error);
	}
	
	private static void DrawNeuronBias2(float x, float y, float d, float value, float error, float bias){
		value = Math.max(Math.min(1f, value), 0.0f);
		error = Math.max(Math.min(1f, error), -1.0f);
		d = 15f;
		gc.setFill(Color.rgb(0, (int)(value * 255f), (int)(200 - (value * 0.6))));
		
		gc.fillOval( x - d/2f, y - d/2f, d, d);
		
		// Draw bias bar
		gc.setStroke(Color.rgb(120, 255, 120));
		gc.setLineWidth(3f + (bias * 2f));
		gc.strokeOval( x - d/2f, y - d/2f, d, d);
		
		// Draw error bar
		error *= 5f;
		gc.setFill(Color.rgb(255, 0,0));
		gc.fillRect(
				x - d / 6f, 
				y + (error > 0.0 ? - error * 2f :  0.0),  
				d / 3f, 
				Math.abs(error) * 2f);
		//System.out.println(error);
		//pl("error: " + error);
	}
	
	
	
	private static void DrawSynapse(double x1, double y1, double x2, double y2, double weight, double thickness, double alpha){
//		weight = Math.max(Math.min(255, weight), 0);
//		gc.setStroke(Color.rgb(255 - weight, weight, 0));
		if(weight >= 0)
			gc.setStroke(Color.GREENYELLOW.interpolate(Color.LIMEGREEN, weight * 2.0));
		else
			gc.setStroke(Color.ORANGE.interpolate(Color.RED, -weight * 2.0));
		//gc.setStroke(Color.RED.interpolate(Color.GREEN, weight * 10.0 + 5.0));
		
		gc.setGlobalAlpha(alpha * 2.0 + 0.02);

		gc.setLineWidth(Math.min(100, thickness));
		//gc.setLineWidth(Math.sqrt(thickness));
		gc.strokeLine(x1, y1, x2, y2);
		gc.setGlobalAlpha(1.0);
	}
	
	public static int truncate(int x, int min, int max){
		return Math.min(max, Math.max(min, x)); 
	}



    public static void line(){System.out.println(); }
    public static void pl(Object o){System.out.println(o); }
    public static void p(Object o){System.out.print(o); }
    public static int randomInt(int max){return (int)(Math.random() * max);}
	
}
