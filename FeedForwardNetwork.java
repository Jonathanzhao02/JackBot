public class FeedForwardNetwork extends GenericNetwork{

	public FeedForwardNetwork(int[] architecture, int inputs, ActivationFunction activationFunction){
		layers = new NeuralLayer[architecture.length];
		layers[0] = new NeuralLayer(inputs, architecture[0], activationFunction);

		for(int i = 1; i < architecture.length; i++){
			layers[i] = new NeuralLayer(architecture[i - 1], architecture[i], activationFunction);
		}

	}

	public FeedForwardNetwork(double[][][] weights, double[][] biases, ActivationFunction activationFunction){

		if(weights.length != biases.length){
			throw new RuntimeException("Dimensions mismatch between weights and biases!");
		}

		layers = new NeuralLayer[weights.length];

		for(int i = 0; i < weights.length; i++){
			layers[i] = new NeuralLayer(weights[i], biases[i], activationFunction);
		}

	}

}