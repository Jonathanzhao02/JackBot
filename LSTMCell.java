public class LSTMCell extends NeuralLayer{
	private NodeLayer[] layers = new NodeLayer[4];
	private double[] cellState;
	private double[] hiddenState;

	public LSTMCell(int inputs, int outputs){
		this.cellState = new double[outputs];
		this.hiddenState = new double[outputs];
		layers[0] = new NodeLayer(inputs + outputs, outputs, ActivationFunction.SIGMOID);
		layers[1] = new NodeLayer(inputs + outputs, outputs, ActivationFunction.SIGMOID);
		layers[2] = new NodeLayer(inputs + outputs, outputs, ActivationFunction.TANH);
		layers[3] = new NodeLayer(inputs + outputs, outputs, ActivationFunction.SIGMOID);
	}

	@Override
	public double[] passThru(double[] inputs){
		this.inputs = inputs.clone();
		double[] combinedState = MatrixOps.concatenate(inputs, hiddenState);
		double[] forgetState = layers[0].passThru(combinedState);
		double[] rememberState = MatrixOps.pointwiseMult(layers[1].passThru(combinedState), layers[2].passThru(combinedState));

		cellState = MatrixOps.pointwiseMult(cellState, forgetState);
		cellState = MatrixOps.pointwiseAdd(cellState, rememberState);

		outputs = MatrixOps.pointwiseMult(layers[3].passThru(combinedState), MatrixOps.pointwiseTanh(cellState));
		hiddenState = outputs.clone();
		return outputs.clone();
	}

}