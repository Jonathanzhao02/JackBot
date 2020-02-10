import java.util.*;

public class Test{
	
	public static void main(String[] args){
		NeuralLayer.setRandom(new Random());
		LSTMCell test = new LSTMCell(5, 2);

		double[] inputs = {1, 0.5, 0.25, 0.125, 0.0625};

		for(int i = 0; i < 20; i++){
			double[] output = test.passThru(inputs);
			System.out.println(output[0] + " " + output[1]);
		}

	}

}