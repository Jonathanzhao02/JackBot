public class NeuralLayer{
    protected double[] outputs;
    protected double[] inputs;
    
    public NeuralLayer(){}

    public double[] passThru(double[] inputs){return inputs;}

    public double[] getUnactivatedOutputs(){return outputs;}
    public double[] getActivatedOutputs(){return outputs;}
}