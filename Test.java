import java.util.*;

public class Test{
	private static Scanner scanner = new Scanner(System.in);
	
	public static void main(String[] args){
		NodeLayer.setRandom(new Random());
		
		int[] architecture = {8, 8, 1};
		int inputs = 1;
		ActivationFunction[] activationFunctions = {ActivationFunction.LEAKY_RELU, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID};
		LossFunction lossFunction = LossFunction.MEAN_SQUARE;

		FeedForwardNetwork test = new FeedForwardNetwork(architecture, inputs, activationFunctions, lossFunction);

		double[][] in = new double[1000][];
		double[][] labels = new double[1000][];

		for(int i = 0; i < in.length; i++){

			double check = (double) i / (double) in.length;

			double[] thing = {check};
			double[] thing2 = {check * check};

			in[i] = thing;
			labels[i] = thing2;
		}

//		while(averageLoss(test, in, labels) > 0.0001){
//			System.out.println(averageLoss(test, in, labels));
//			test.stochasticGradientDescentWithMomentum(in, labels);
//		}

		for(int i = 1; i <= 10000; i++){

			if(i % 1000 == 0 || i == 1){
				System.out.println("Epoch " + i);
				System.out.println(averageLoss(test, in, labels));
			}

			test.stochasticGradientDescentWithNAG(in, labels);
			shuffleData(in, labels);
		}

		System.out.println(averageLoss(test, in, labels));
		test.printWeights();

		while(true){
			double[] inp = {scanner.nextDouble()};
			double[] outputs = test.passThru(inp);

			for(int i = 0; i < outputs.length; i++){
				System.out.println("Output" + i + ": " + outputs[i]);
			}

		}

	}

	private static void shuffleData(double[][] in, double[][] labels){
		Random rand = new Random();
		double[] temp;
		int index;

		for(int i = 0; i < in.length; i++){
			index = rand.nextInt(in.length);

			temp = in[index];
			in[index] = in[0];
			in[0] = temp;

			temp = labels[index];
			labels[index] = labels[0];
			labels[0] = temp;
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