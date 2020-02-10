public class GenericNetwork{
	protected Object[] layers;

	public GenericNetwork(Object[] layers){
		this.layers = layers;
	}

	public GenericNetwork(){

	}

	public double[] passThru(double[] inputs){
		double[] outputs = inputs;

		for(int i = 0; i < layers.length; i++){

			//ADD MORE HERE AS MORE CLASSES ARE MADE
			if(layers[i].getClass() == LSTMCell.class){
				outputs = ((LSTMCell) layers[i]).passThru(outputs);
			} else if(layers[i].getClass() == NeuralLayer.class){
				outputs = ((NeuralLayer) layers[i]).passThru(outputs);
			}

		}

		return outputs;
	}

	public Object[] getLayers(){return layers;}
}