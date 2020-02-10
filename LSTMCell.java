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

	public double[] passThru(double[] inputs){
		double[] combinedState = MatrixOps.concatenate(inputs, hiddenState);
		double[] forgetState = layers[0].passThru(combinedState);
		double[] rememberState = MatrixOps.pointwiseMult(layers[1].passThru(combinedState), layers[2].passThru(combinedState));

		cellState = MatrixOps.pointwiseMult(cellState, forgetState);
		cellState = MatrixOps.pointwiseAdd(cellState, rememberState);

		double[] outputs = MatrixOps.pointwiseMult(layers[3].passThru(combinedState), MatrixOps.pointwiseTanh(cellState));
		hiddenState = outputs.clone();
		return outputs;
	}

}