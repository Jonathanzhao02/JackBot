import java.util.*;

public class Test{
	private static Scanner scanner = new Scanner(System.in);
	
	public static void main(String[] args){
		NodeLayer.setRandom(new Random());
		
		int[] architecture = {3, 3, 1};
		int inputs = 1;
		ActivationFunction[] activationFunctions = {ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU};
		LossFunction lossFunction = LossFunction.MEAN_SQUARE;

		FeedForwardNetwork test = new FeedForwardNetwork(architecture, inputs, activationFunctions, lossFunction);

		double[][] in = new double[10000][];
		double[][] labels = new double[10000][];

		for(int i = 0; i < in.length; i++){

			double[] thing = {i / 10000.0};
			double[] thing2 = {i * i / 10000.0 / 10000.0};

			in[i] = thing;
			labels[i] = thing2;
		}

//		while(averageLoss(test, in, labels) > 0.0001){
//			System.out.println(averageLoss(test, in, labels));
//			test.stochasticGradientDescentWithMomentum(in, labels);
//		}

		for(int i = 0; i < 500; i++){
			System.out.println(averageLoss(test, in, labels));
			test.stochasticGradientDescentWithMomentum(in, labels);
		}

		System.out.println(averageLoss(test, in, labels));

		while(true){
			double[] inp = {scanner.nextDouble()};
			double[] outputs = test.passThru(inp);

			for(int i = 0; i < outputs.length; i++){
				System.out.println("Output" + i + ": " + outputs[i]);
			}

		}

	}

	private static double averageLoss(FeedForwardNetwork test, double[][] inputs, double[][] labels){
		double loss = 0;

		for(int i = 0; i < inputs.length; i++){
			loss += test.getLoss(inputs[i], labels[i]);
		}

		return loss / (double) inputs.length;
	}

}