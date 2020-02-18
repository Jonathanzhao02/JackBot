public class FeedForwardNetwork extends GenericNetwork{

	public FeedForwardNetwork(int[] architecture, int inputs, ActivationFunction[] activationFunctions, LossFunction lossFunction){
		layers = new NodeLayer[architecture.length];
		layers[0] = new NodeLayer(inputs, architecture[0], activationFunctions[0]);

		if(architecture.length > 1){

			for(int i = 1; i < architecture.length; i++){
				layers[i] = new NodeLayer(architecture[i - 1], architecture[i], activationFunctions[i]);
			}

		}

		this.lossFunction = lossFunction;
	}

	public FeedForwardNetwork(double[][][] weights, double[][] biases, ActivationFunction[] activationFunctions, LossFunction lossFunction){

		if(weights.length != biases.length){
			throw new RuntimeException("Dimensions mismatch between weights " + weights.length + " and biases " + biases.length + "!");
		}

		layers = new NodeLayer[weights.length];

		for(int i = 0; i < weights.length; i++){
			layers[i] = new NodeLayer(weights[i], biases[i], activationFunctions[i]);
		}

		this.lossFunction = lossFunction;
	}

}