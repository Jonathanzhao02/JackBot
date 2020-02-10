public class LSTMCell{
	private NeuralLayer[] layers = new NeuralLayer[4];
	private double[] cellState;
	private double[] hiddenState;

	public LSTMCell(int inputs, int outputs){
		this.cellState = new double[outputs];
		this.hiddenState = new double[outputs];
		layers[0] = new NeuralLayer(inputs + outputs, outputs, ActivationFunction.Sigmoid);
		layers[1] = new NeuralLayer(inputs + outputs, outputs, ActivationFunction.Sigmoid);
		layers[2] = new NeuralLayer(inputs + outputs, outputs, ActivationFunction.Tanh);
		layers[3] = new NeuralLayer(inputs + outputs, outputs, ActivationFunction.Sigmoid);
	}

	private double[] concatenate(double[] vec1, double[] vec2){
		double[] newVec = new double[vec1.length + vec2.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i];
		}

		for(int i = vec1.length; i < vec1.length + vec2.length; i++){
			newVec[i] = vec2[i - vec1.length];
		}

		return newVec;
	}

	private double[] pointwiseMult(double[] vec1, double[] vec2){

		if(vec1.length != vec2.length){
			throw new RuntimeException("Dimension mismatch between vectors!");
		}

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] * vec2[i];
		}

		return newVec;
	}

	private double[] pointwiseTanh(double[] vec){
		double[] newVec = new double[vec.length];

		for(int i = 0; i < vec.length; i++){
			newVec[i] = Math.tanh(vec[i]);
		}

		return newVec;
	}

	private double[] pointwiseAdd(double[] vec1, double[] vec2){

		if(vec1.length != vec2.length){
			throw new RuntimeException("Dimension mismatch between vectors!");
		}

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] + vec2[i];
		}

		return newVec;
	}

	public double[] passThru(double[] inputs){
		double[] combinedState = concatenate(inputs, hiddenState);
		double[] forgetState = layers[0].passThru(combinedState);
		double[] rememberState = pointwiseMult(layers[1].passThru(combinedState), layers[2].passThru(combinedState));

		cellState = pointwiseMult(cellState, forgetState);
		cellState = pointwiseAdd(cellState, rememberState);

		double[] outputs = pointwiseMult(layers[3].passThru(combinedState), pointwiseTanh(cellState));
		hiddenState = outputs.clone();
		return outputs;
	}

}