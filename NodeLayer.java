import java.util.Random;

public class NodeLayer extends NeuralLayer{

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
		NodeLayer.rand = rand;
	}

	public void setActivation(ActivationFunction activationFunction){
		this.activationFunction = activationFunction;
	}

	public NodeLayer(int inputs, int outputs, ActivationFunction activationFunction){
		genWeights(inputs, outputs);
		genBiases(outputs);
		this.activationFunction = activationFunction;
	}

	public NodeLayer(double[][] weights, double[] biases, ActivationFunction activationFunction){
		this.weights = weights;
		this.biases = biases;

		if(biases.length != weights.length){
			throw new RuntimeException("Dimension mismatch between biases " + biases.length + " and weights " + weights.length + "!");
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

	@Override
	public double[] passThru(double[] inputs){

		if(inputs.length != weights[0].length){
			throw new RuntimeException("Dimension mismatch between inputs " + inputs.length + " and weights " + weights[0].length + "!");
		}

		this.inputs = inputs.clone();
        outputs = MatrixOps.matrixMult(weights, inputs);
		outputs = MatrixOps.pointwiseAdd(outputs, biases);

		return MatrixOps.pointwiseFunction(outputs, activationFunction);
	}

	private double ReLUPrime(double x){
		
		if(x > 0){
			return 1;
		} else{
			return GenericNetwork.LEAKY_SLOPE;
		}

	}

	private double leakyReLUPrime(double x){
		
		if(x > 0){
			return 1;
		} else{
			return GenericNetwork.LEAKY_SLOPE;
		}

	}

	private double sigmoidPrime(double x){
		return (1.0 / (1.0 + Math.pow(Math.E, -x))) * (1 - 1.0 / (1.0 + Math.pow(Math.E, -x)));
	}

	public Object[] backPropagate(double[] gradient){
		Object[] results = new Object[3];
		double[] derivativeMat = new double[outputs.length];

		for(int i = 0; i < derivativeMat.length; i++){

			switch(activationFunction){
				case RELU:
					derivativeMat[i] = ReLUPrime(outputs[i]);
					break;
				case LEAKY_RELU:
					derivativeMat[i] = leakyReLUPrime(outputs[i]);
					break;
				case SIGMOID:
					derivativeMat[i] = sigmoidPrime(outputs[i]);
					break;
				case TANH:
					break;
				case SOFTMAX:
					break;
				default:
					throw new RuntimeException("No implementation found for activation function " + activationFunction.name());
			}

		}

		double[] gradientMat = MatrixOps.pointwiseMult(derivativeMat, gradient);
		results[0] = gradientMat;
		results[1] = MatrixOps.matrixMult(gradientMat, inputs);
		return results;
	}

	public void adjustWeights(double[][] weights){

		if(weights.length != this.weights.length || weights[0].length != this.weights[0].length){
			throw new RuntimeException("Gradient dims " + weights.length + "x" + weights[0].length + " does not match weights dims " + this.weights.length + "x" + this.weights[0].length + "!");
		}

		this.weights = MatrixOps.pointwiseSubtract(this.weights, MatrixOps.scalarMult(weights, GenericNetwork.LEARNING_RATE));

		if(GenericNetwork.WEIGHT_DECAY < 1){
			this.weights = MatrixOps.scalarMult(this.weights, GenericNetwork.WEIGHT_DECAY);
		}

	}

	public void adjustBiases(double[] biases){

		if(biases.length != this.biases.length){
			throw new RuntimeException("Gradient length " + biases.length + " does not match biases length " + this.biases.length + "!");
		}

		this.biases = MatrixOps.pointwiseSubtract(this.biases, MatrixOps.scalarMult(biases, GenericNetwork.LEARNING_RATE));

		if(GenericNetwork.WEIGHT_DECAY < 1){
			this.biases = MatrixOps.scalarMult(this.biases, GenericNetwork.WEIGHT_DECAY);
		}

	}

	public double[][] getWeights(){return weights;}
	public double[] getBiases(){return biases;}

	@Override
	public double[] getActivatedOutputs(){return MatrixOps.pointwiseFunction(outputs, activationFunction);}
}