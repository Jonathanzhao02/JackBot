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

	private Object[] backPropagate(double[] inputs, double[] labels){
		passThru(inputs);
		double[] lossGradient = getLossDerivatives(layers[layers.length - 1].getActivatedOutputs(), labels);
		double[][][] weightAdjustments = new double[layers.length][][];
		double[][] biasAdjustments = new double[layers.length][];

		for(int i = layers.length - 1; i >= 0; i--){
			NodeLayer layer = (NodeLayer) layers[i];
			Object[] propagationResults = layer.backPropagate(lossGradient);
			lossGradient = (double[]) propagationResults[0];
			weightAdjustments[i] = (double[][]) propagationResults[1];
			biasAdjustments[i] = lossGradient.clone();
			lossGradient = MatrixOps.matrixMult(MatrixOps.transpose(layer.getWeights()), lossGradient);
		}

		Object[] results = {weightAdjustments, biasAdjustments};
		return results;
	}

	private Object[] nesterovBackPropagate(double[] inputs, double[] labels, double[][] prevGradient){
		passThru(inputs);
		double[] lossGradient = getLossDerivatives(layers[layers.length - 1].getActivatedOutputs(), labels);
		double[][][] weightAdjustments = new double[layers.length][][];
		double[][] biasAdjustments = new double[layers.length][];

		for(int i = layers.length - 1; i >= 0; i--){
			NodeLayer layer = (NodeLayer) layers[i];
			Object[] propagationResults = layer.backPropagate(MatrixOps.pointwiseSubtract(lossGradient, prevGradient[i]));
			lossGradient = (double[]) propagationResults[0];
			weightAdjustments[i] = (double[][]) propagationResults[1];
			biasAdjustments[i] = lossGradient.clone();
			prevGradient[i] = biasAdjustments[i];
			lossGradient = MatrixOps.matrixMult(MatrixOps.transpose(layer.getWeights()), lossGradient);
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
				weightAdjustments = MatrixOps.pointwiseAdd((double[][][]) results[0], MatrixOps.scalarMult(weightAdjustments, MOMENTUM));
				biasAdjustments = MatrixOps.pointwiseAdd((double[][]) results[1], MatrixOps.scalarMult(biasAdjustments, MOMENTUM));
			}

			adjustLayers(weightAdjustments, biasAdjustments);
		}

	}

	public void stochasticGradientDescentWithNAG(double[][] inputs, double[][] labels){
		Object[] results = null;
		double[][][] weightAdjustments = null;
		double[][] biasAdjustments = null;

		results = backPropagate(inputs[0], labels[0]);
		weightAdjustments = (double[][][]) results[0];
		biasAdjustments = (double[][]) results[1];
		adjustLayers(weightAdjustments, biasAdjustments);

		for(int i = 1; i < inputs.length; i++){
			results = nesterovBackPropagate(inputs[i], labels[i], MatrixOps.scalarMult(biasAdjustments, MOMENTUM));
			weightAdjustments = MatrixOps.pointwiseAdd((double[][][]) results[0], MatrixOps.scalarMult(weightAdjustments, MOMENTUM));
			biasAdjustments = MatrixOps.pointwiseAdd((double[][]) results[1], MatrixOps.scalarMult(biasAdjustments, MOMENTUM));
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

	public void miniBatchGradientDescent(double[][] inputs, double[][] labels, int batchSize){
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

	public void miniBatchGradientDescentWithMomentum(double[][] inputs, double[][] labels, int batchSize){
		Object[] results = null;
		double[][][] avgWeightAdjustments = null;
		double[][] avgBiasAdjustments = null;
		double[][][] prevWeightAdjustments = null;
		double[][] prevBiasAdjustments = null;
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

			if(prevWeightAdjustments != null){
				avgWeightAdjustments = MatrixOps.pointwiseAdd(avgWeightAdjustments, MatrixOps.scalarMult(prevWeightAdjustments, MOMENTUM));
				avgBiasAdjustments = MatrixOps.pointwiseAdd(avgBiasAdjustments, MatrixOps.scalarMult(prevBiasAdjustments, MOMENTUM));
			}

			prevWeightAdjustments = avgWeightAdjustments;
			prevBiasAdjustments = avgBiasAdjustments;
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

			if(prevWeightAdjustments != null){
				avgWeightAdjustments = MatrixOps.pointwiseAdd(avgWeightAdjustments, MatrixOps.scalarMult(prevWeightAdjustments, MOMENTUM));
				avgBiasAdjustments = MatrixOps.pointwiseAdd(avgBiasAdjustments, MatrixOps.scalarMult(prevBiasAdjustments, MOMENTUM));
			}

			prevWeightAdjustments = avgWeightAdjustments;
			prevBiasAdjustments = avgBiasAdjustments;
			adjustLayers(avgWeightAdjustments, avgBiasAdjustments);
			avgWeightAdjustments = null;
			avgBiasAdjustments = null;
		}

	}

	private void adjustLayers(double[][][] weightAdjustments, double[][] biasAdjustments){

		for(int i = 0; i < layers.length; i++){
			NodeLayer layer = (NodeLayer) layers[i];
			layer.adjustWeights(weightAdjustments[i]);
			layer.adjustBiases(biasAdjustments[i]);
		}

	}

	public void printWeights(){

		for(int i = 0; i < layers.length; i++){
			NodeLayer layer = (NodeLayer) layers[i];
			double[][] wghts = layer.getWeights();
			double[] bses = layer.getBiases();

			for(int j = 0; j < wghts.length; j++){

				for(int k = 0; k < wghts[j].length; k++){
					System.out.println("Layer" + i + " weight " + j + ", " + k + ": " + wghts[j][k]);
				}

			}

			for(int j = 0; j < bses.length; j++){
				System.out.println("Layer" + i + " bias " + j + ": " + bses[j]);
			}

		}

	}

}