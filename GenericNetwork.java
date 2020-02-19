public class GenericNetwork{
	public static double LEARNING_RATE = 0.1;
	public static double LEAKY_SLOPE = 0.01;
	public static double MOMENTUM = 0.9;
	public static double WEIGHT_DECAY = 1;
	protected NeuralLayer[] layers;
	protected LossFunction lossFunction;

	public GenericNetwork(NeuralLayer[] layers, LossFunction lossFunction){
		this.layers = layers;
		this.lossFunction = lossFunction;
	}

	public GenericNetwork(){

	}

	public double[] passThru(double[] inputs){

        if(layers == null){
            throw new RuntimeException("No layers set!");
        }

		double[] outputs = inputs;

		for(int i = 0; i < layers.length; i++){
			outputs = layers[i].passThru(outputs);
		}

		return outputs;
	}

	public double calcMeanSquareLoss(double[] labels, double[] outputs){
		double loss = 0;

		for(int i = 0; i < outputs.length; i++){
			loss += (labels[i] - outputs[i]) * (labels[i] - outputs[i]);
		}

		loss /= (double) outputs.length;
		return loss;
	}

	public double calcCrossEntropyLoss(double[] labels, double[] outputs){
		double loss = 0;

		for(int i = 0; i < outputs.length; i++){
			loss -= labels[i] * Math.log(outputs[i]);
		}

		return loss;
	}

	public double getLoss(double[] inputs, double[] labels){
		passThru(inputs);
		double[] outputs = layers[layers.length - 1].getActivatedOutputs();

		if(labels.length != outputs.length){
			throw new RuntimeException("Labeled outputs length " + labels.length + " does not match network output length " + outputs.length + "!");
		}

		switch(lossFunction){
			case MEAN_SQUARE:
				return calcMeanSquareLoss(labels, outputs);
			case CROSS_ENTROPY:
				return calcCrossEntropyLoss(labels, outputs);
			default:
				throw new RuntimeException("No loss function implementation found for " + lossFunction.name());
		}

	}

	private double meanSquarePrime(double output, double label){
		return output - label;
	}

	protected double[] getLossDerivatives(double[] outputs, double[] labels){
		double[] derivatives = new double[outputs.length];

		for(int i = 0; i < outputs.length; i++){
			//ASSUMES USING MEAN SQUARE LOSS
			derivatives[i] = meanSquarePrime(outputs[i], labels[i]) / (double) outputs.length;
		}

		return derivatives;
	}

	public Object[] getLayers(){return layers;}
}