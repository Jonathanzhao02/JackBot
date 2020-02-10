import java.util.Random;

public class NeuralLayer{

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

	public void setActivation(ActivationFunction activationFunction){
		this.activationFunction = activationFunction;
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

        double[] outputs = MatrixOps.matrixMult(weights, inputs);

        for(int i = 0; i < weights.length; i++){
            outputs[i] += biases[i];

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