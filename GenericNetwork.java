public class GenericNetwork{
	public static double GRADIENT_DELTA = 0.1;
	public static double LEAKY_SLOPE = 0.0001;
	public static double MOMENTUM = 0.01;
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

	private double[] getLossDerivatives(double[] outputs, double[] labels){
		double[] derivatives = new double[outputs.length];

		for(int i = 0; i < outputs.length; i++){
			//ASSUMES USING MEAN SQUARE LOSS
			derivatives[i] = meanSquarePrime(outputs[i], labels[i]) / (double) outputs.length;
		}

		return derivatives;
	}

	private Object[] backPropagate(double[] inputs, double[] labels){
		//ASSUMES USING FEEDFORWARD
		passThru(inputs);
		double[] lossGradient = getLossDerivatives(layers[layers.length - 1].getActivatedOutputs(), labels);
		double[][][] weightAdjustments = new double[layers.length][][];
		double[][] biasAdjustments = new double[layers.length][];

		for(int i = layers.length - 1; i >= 0; i--){

			//REPLACE WITH BACKPROPAGATE CODE
			if(layers[i].getClass() == NodeLayer.class){
				NodeLayer layer = (NodeLayer) layers[i];
				Object[] propagationResults = layer.backPropagate(lossGradient);
				lossGradient = (double[]) propagationResults[0];
				weightAdjustments[i] = (double[][]) propagationResults[1];
				biasAdjustments[i] = lossGradient.clone();
				lossGradient = MatrixOps.matrixMult(MatrixOps.transpose(layer.getWeights()), lossGradient);
			}

		}

		Object[] results = {weightAdjustments, biasAdjustments};
		return results;
	}

	public void stochasticGradientDescentWithMomentum(double[][] inputs, double[][] labels){
		Object[] results = null;
		double[][][] weightAdjustments = null;
		double[][] biasAdjustments = null;

		for(int i = 0; i < inputs.length; i++){
			results = backPropagate(inputs[i], labels[i]);

			if(weightAdjustments == null){
				weightAdjustments = (double[][][]) results[0];
				biasAdjustments = (double[][]) results[1];
			} else{
				weightAdjustments = MatrixOps.pointwiseAdd(MatrixOps.scalarMult((double[][][]) results[0], 1.0 - MOMENTUM), MatrixOps.scalarMult(weightAdjustments, MOMENTUM));
				biasAdjustments = MatrixOps.pointwiseAdd(MatrixOps.scalarMult((double[][]) results[1], 1.0 - MOMENTUM), MatrixOps.scalarMult(biasAdjustments, MOMENTUM));
			}

			adjustLayers(weightAdjustments, biasAdjustments);
		}

	}

	public void stochasticGradientDescent(double[][] inputs, double[][] labels){
		Object[] results = null;

		for(int i = 0; i < inputs.length; i++){
			results = backPropagate(inputs[i], labels[i]);

			double[][][] weightAdjustments = (double[][][]) results[0];
			double[][] biasAdjustments = (double[][]) results[1];

			adjustLayers(weightAdjustments, biasAdjustments);
		}

	}

	public void batchGradientDescent(double[][] inputs, double[][] labels, int batchSize){
		Object[] results = null;
		double[][][] avgWeightAdjustments = null;
		double[][] avgBiasAdjustments = null;
		int remainder = inputs.length % batchSize;

		for(int i = 0; i < inputs.length / batchSize; i++){

			for(int j = 0; j < batchSize; j++){
				results = backPropagate(inputs[j], labels[j]);
				
				if(avgWeightAdjustments == null){
					avgWeightAdjustments = (double[][][]) results[0];
					avgBiasAdjustments = (double[][]) results[1];
				} else{
					avgWeightAdjustments = MatrixOps.pointwiseAdd(avgWeightAdjustments, (double[][][]) results[0]);
					avgBiasAdjustments = MatrixOps.pointwiseAdd(avgBiasAdjustments, (double[][]) results[1]);
				}

			}

			avgWeightAdjustments = MatrixOps.scalarMult(avgWeightAdjustments, 1.0 / (double) batchSize);
			avgBiasAdjustments = MatrixOps.scalarMult(avgBiasAdjustments, 1.0 / (double) batchSize);
			adjustLayers(avgWeightAdjustments, avgBiasAdjustments);
			avgWeightAdjustments = null;
			avgBiasAdjustments = null;
		}

		if(inputs.length % batchSize != 0){

			for(int i = 1; i <= remainder; i++){
				results = backPropagate(inputs[inputs.length - i], labels[inputs.length - i]);
				
				if(avgWeightAdjustments == null){
					avgWeightAdjustments = (double[][][]) results[0];
					avgBiasAdjustments = (double[][]) results[1];
				} else{
					avgWeightAdjustments = MatrixOps.pointwiseAdd(avgWeightAdjustments, (double[][][]) results[0]);
					avgBiasAdjustments = MatrixOps.pointwiseAdd(avgBiasAdjustments, (double[][]) results[1]);
				}

			}

			avgWeightAdjustments = MatrixOps.scalarMult(avgWeightAdjustments, 1.0 / (double) remainder);
			avgBiasAdjustments = MatrixOps.scalarMult(avgBiasAdjustments, 1.0 / (double) remainder);
			adjustLayers(avgWeightAdjustments, avgBiasAdjustments);
			avgWeightAdjustments = null;
			avgBiasAdjustments = null;

		}

	}

	private void adjustLayers(double[][][] weightAdjustments, double[][] biasAdjustments){

		for(int i = 0; i < layers.length; i++){

			if(layers[i].getClass() == NodeLayer.class){
				NodeLayer layer = (NodeLayer) layers[i];
				layer.adjustWeights(weightAdjustments[i]);
				layer.adjustBiases(biasAdjustments[i]);
			}

		}

	}

	private void print(double[][] one, double[] two){

		for(int i = 0; i < one.length; i++){

			for(int j = 0; j < one[0].length; j++){
				System.out.println("Weight " + i + ", " + j + ": " + one[i][j]);
			}

		}

		for(int i = 0; i < two.length; i++){
			System.out.println("Bias " + i + ": " + two[i]);
		}

	}

	public Object[] getLayers(){return layers;}
}