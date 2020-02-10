import java.util.*;

public class Test{
	
	public static void main(String[] args){
		NeuralLayer.setRandom(new Random());
		LSTMCell test = new LSTMCell(5, 2);

		double[] inputs = {1, 0.5, 0.25, 0.125, 0.0625};

		for(int i = 0; i < 20; i++){
			double[] output = test.passThru(inputs);
			System.out.println(output[0] + " " + output[1]);
		}

	}

}

class RecurrentNetwork{
	private Object[] layers;

	public RecurrentNetwork(Object[] layers){
		this.layers = layers;
	}

}

class FeedForwardNeuralNetwork{
	private NeuralLayer[] layers;

	private boolean layersMatch(){
		int numInputs;
		int numOutputs = layers[0].getBiases().length;
		boolean match = true;

		for(int i = 1; i < layers.length; i++){
			numOutputs = layers[i - 1].getBiases().length;
			numInputs = layers[i].getWeights()[0].length;

			if(numOutputs != numInputs){
				match = false;
			}

		}

		return match;
	}

	public FeedForwardNeuralNetwork(NeuralLayer[] layers){
		this.layers = layers;

		if(!layersMatch()){
			throw new RuntimeException("Dimension mismatch between layers!");
		}

	}

	public FeedForwardNeuralNetwork(int[] architecture, int inputs, ActivationFunction activationFunction){
		layers = new NeuralLayer[architecture.length];
		layers[0] = new NeuralLayer(inputs, architecture[0], activationFunction);

		for(int i = 1; i < architecture.length; i++){
			layers[i] = new NeuralLayer(architecture[i - 1], architecture[i], activationFunction);
		}

	}

	public FeedForwardNeuralNetwork(double[][][] weights, double[][] biases, ActivationFunction activationFunction){

		if(weights.length != biases.length){
			throw new RuntimeException("Dimensions mismatch between weights and biases!");
		}

		layers = new NeuralLayer[weights.length];

		for(int i = 0; i < weights.length; i++){
			layers[i] = new NeuralLayer(weights[i], biases[i], activationFunction);
		}

		if(!layersMatch()){
			throw new RuntimeException("Dimension mismatch between layers!");
		}

	}

	public double[] passThru(double[] inputs){
		double[] outputs = inputs;

		for(int i = 0; i < layers.length; i++){
			outputs = layers[i].passThru(outputs);
		}

		return outputs;
	}

}

class LSTMCell{
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

class NeuralLayer{

	//FIRST WEIGHTS DIMENSION: OUTPUT NUMBER
	//SECOND WEIGHTS DIMENSION: INPUT NUMBER

	//BIAS DIMESION: OUTPUT NUMBER

	//FIRST HIDDEN WIEGHTS DIMENSION: INPUT NUMBER
	//SECOND HIDDEN WEIGHTS DIMENSION: OUTPUT NUMBER

	private static Random rand;
	private double[][] weights;
	private double[] biases;
	private ActivationFunction activationFunction;

	public static void setRandom(Random rand){
		NeuralLayer.rand = rand;
	}

	public NeuralLayer(int inputs, int outputs, ActivationFunction activationFunction){
		genWeights(inputs, outputs);
		genBiases(outputs);
		this.activationFunction = activationFunction;
	}

	public NeuralLayer(double[][] weights, double[] biases, ActivationFunction activationFunction){
		this.weights = weights;
		this.biases = biases;

		if(biases.length != weights.length){
			throw new RuntimeException("Dimension mismatch between biases and weights!");
		}

		this.activationFunction = activationFunction;
	}

	public void genWeights(int inputs, int outputs){
		weights = new double[outputs][inputs];

		for(int i = 0; i < outputs; i++){

			for(int j = 0; j < inputs; j++){
				weights[i][j] = rand.nextGaussian();
			}

		}

	}

	public void genBiases(int outputs){
		biases = new double[outputs];

		for(int i = 0; i < outputs; i++){
			biases[i] = rand.nextGaussian();
		}

	}

	private double ReLU(double input){
		
		if(input > 0){
			return input;
		} else{
			return 0;
		}

	}

	private double sigmoid(double input){
		return 1 / (1 + Math.pow(Math.E, -input));
	}

	private double tanh(double input){
		return Math.tanh(input);
	}

	public double[] passThru(double[] inputs){

		if(inputs.length != weights[0].length){
			throw new RuntimeException("Dimension mismatch between inputs and weights!");
		}

		double[] outputs = new double[weights.length];

		for(int i = 0; i < weights.length; i++){

			for(int j = 0; j < inputs.length; j++){
				outputs[i] += weights[i][j] * inputs[j];
			}

			outputs[i] += biases[i];
			outputs[i] = ReLU(outputs[i]);

			switch(activationFunction){
				case ReLU:
					outputs[i] = ReLU(outputs[i]);
					break;
				case Sigmoid:
					outputs[i] = sigmoid(outputs[i]);
					break;
				case Tanh:
					outputs[i] = tanh(outputs[i]);
					break;
			}

		}

		return outputs;
	}

	public double[][] getWeights(){return weights;}
	public double[] getBiases(){return biases;}
}

enum ActivationFunction{
	ReLU,
	Sigmoid,
	Tanh,
}