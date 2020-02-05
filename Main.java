import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;

import javafx.application.Application;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.geometry.Pos;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.control.Label;
import javafx.animation.Timeline;
import javafx.animation.KeyFrame;
import javafx.util.Duration;

public class Main extends Application{
	private Random rand = new Random();

	private int populationSize = 1000;
	private double mutationChance = 0.05;
	private double mutationMagnitude = 0.1;
	private double crossoverChance = 0.5;
	private double elitePortion = 0.1;
	private double maxWeightMagnitude = 5;

	private final int numDecks = 1;

	private int[] architecture = {26, 26, 26, 2};
	private int inputs = 14;

	private LinkedBlockingQueue<Child> bestChildren = new LinkedBlockingQueue<Child>();
	private volatile Child bestChild;
	private volatile boolean finishedGame = true;

	Dealer g_dealer;
	Player g_player;
	double[] g_outputs = null;

	public static void main(String[] args){
		launch(args);
	}

	public void start(Stage mainStage){
		Player.setRandom(rand);
		Generation.setRandom(rand);
		Child.setRandom(rand);
		NeuralNetwork.setRandom(rand);
		NeuralLayer.setRandom(rand);
		Generation.setParameters(populationSize, mutationChance, mutationMagnitude, maxWeightMagnitude, crossoverChance, elitePortion);

		g_dealer = new Dealer(numDecks);
		g_player = g_dealer.getPlayer();

		Label dealerHand = new Label();
		Label playerHand = new Label();

		dealerHand.setAlignment(Pos.CENTER);
		playerHand.setAlignment(Pos.CENTER);

		BorderPane rootNode = new BorderPane();
		rootNode.setTop(dealerHand);
		rootNode.setBottom(playerHand);
		Scene mainScene =  new Scene(rootNode, 400, 400);
		mainStage.setScene(mainScene);
		mainStage.setTitle("JackBot");
		mainStage.show();

		Thread trainThread = new Thread(() -> {
			Generation current = new Generation(architecture, inputs);
		
			while(true){

				for(Child child : current.getChildren()){
					playGame(child);
					child.getNetwork().resetHidden();
				}

				System.out.println(current.getBest().getFitness());
				bestChildren.add(current.getBest().clone());
				current.generateNew();
			}

		});

		trainThread.start();

		Timeline showGame = new Timeline(new KeyFrame(Duration.millis(1000), e -> {
			dealerHand.setText("Dealer: " + g_dealer.getHand());
			playerHand.setText("Player: " + g_player.getHand());
			graphicPlayGame();
		}));

		showGame.setOnFinished(e -> {

			if(!finishedGame){
				showGame.play();
			} else{
				dealerHand.setText("Dealer (final): " + g_dealer.getHand());
				playerHand.setText("Player (final): " + g_player.getHand());
				g_player.clearHand();
				g_dealer.clearHand();

				finishedGame = true;
			}

		});

		Thread acceptThread = new Thread(() -> {

			while(true){

				if(finishedGame){

					try{
						bestChild = bestChildren.take();
					} catch(InterruptedException e){
						System.out.println("Thread interrupted!");
					}

					finishedGame = false;

					g_player.shuffledProperty().addListener((e, ol, ne) -> {

						if(ne){
							bestChild.getNetwork().resetHidden();
						}
				
					});
				
					g_dealer.shuffledProperty().addListener((e, ol, ne) -> {
				
						if(ne){
							bestChild.getNetwork().resetHidden();
						}
				
					});

					bestChild.getNetwork().resetHidden();

					for(int i = 0; i < 2; i++){
						g_dealer.hit();
						g_outputs = bestChild.recurrentPassThru(tensorConvert(g_player.hit()));
					}

					showGame.play();
				}

				try{
					Thread.sleep(1000);
				} catch(InterruptedException e){
					System.out.println("Accept thread interrupted!");
				}

			}

		});

		acceptThread.start();
	}

	private void graphicPlayGame(){
		double[] inputs = null;
		g_dealer.hit();

		if(g_outputs[0] > g_outputs[1]){
			inputs = tensorConvert(g_player.hit());
			inputs[13] = g_player.getHand();
			g_outputs = bestChild.recurrentPassThru(inputs);
		} else{
			g_player.stand();
		}

		if(g_player.isStanding() && g_dealer.isStanding()){
			finishedGame = true;
		}

	}

	private void playGame(Child network){
		Dealer dealer = new Dealer(numDecks);
		Player player = dealer.getPlayer();

		player.shuffledProperty().addListener((e, ol, ne) -> {

			if(ne){
				network.getNetwork().resetHidden();
			}

		});

		dealer.shuffledProperty().addListener((e, ol, ne) -> {

			if(ne){
				network.getNetwork().resetHidden();
			}

		});

		network.getNetwork().resetHidden();

		for(int j = 0; j < 100; j++){
			double[] outputs = null;
			double[] inputs = null;

			for(int i = 0; i < 2; i++){
				dealer.hit();
				inputs = tensorConvert(player.hit());
				inputs[13] = player.getHand();
				outputs = network.recurrentPassThru(inputs);
			}

			while(!player.isStanding() || !dealer.isStanding()){
				dealer.hit();
				
				if(outputs[0] > outputs[1]){
					inputs = tensorConvert(player.hit());
					inputs[13] = player.getHand();
					outputs = network.recurrentPassThru(inputs);
				} else{
					player.stand();
				}

			}

			if(player.getHand() > dealer.getHand() && player.getHand() <= 21){
				network.setFitness(network.getFitness() + 1);
			} else if(player.getHand() == dealer.getHand() && player.getHand() <= 21){
				network.setFitness(network.getFitness() + 0.5);
			}

			player.clearHand();
			dealer.clearHand();
		}

	}

	private double[] tensorConvert(int val){
		double[] tensor = new double[14];

		if(val >= 0){
			tensor[val - 1] = 1;
		}

		return tensor;
	}
	
}

class Dealer extends Player{
	private Player player;

	public Dealer(int numDecks){
		super(numDecks);
		player = new Player(numDecks, cards);
	}

	@Override
	public int hit(){

		if(handVal < 17){
			return super.hit();
		} else{
			stand();
			return -1;
		}

	}

	public Player getPlayer(){return player;}
}

class Player{
	protected static Random rand;
	protected int[] hand;
	protected int[] cards;
	protected int numCards = 0;
	protected int handVal = 0;
	protected int numDecks = 0;
	protected boolean standing = false;
	protected SimpleBooleanProperty shuffled = new SimpleBooleanProperty(false);

	public static void setRandom(Random rand){
		Player.rand = rand;
	}

	public Player(int numDecks){
		this.numDecks = numDecks;
		shuffleCards();
		this.hand = new int[21];
	}

	public Player(int numDecks, int[] cards){
		this.numDecks = numDecks;
		this.cards = cards;
		this.hand = new int[21];
	}

	public void stand(){
		standing = true;
	}

	public void clearHand(){
		hand = new int[21];
		handVal = 0;
		numCards = 0;
		standing = false;
	}

	protected void shuffleCards(){
		shuffled.setValue(true);
		shuffled.setValue(false);
		cards = new int[numDecks * 52];
		int randIndex;
		int temp;

		for(int i = 0; i < cards.length; i++){cards[i] = i % 13 + 1;}

		for(int i = 0; i < cards.length; i++){
			randIndex = rand.nextInt(cards.length);
			temp = cards[0];
			cards[0] = cards[randIndex];
			cards[randIndex] = temp;
		}

	}

	private int drawCard(){
		int drawnCard = 0;
		int counter = 0;

		while(drawnCard == 0 && counter < cards.length){
			drawnCard = cards[counter];
			counter++;
		}

		cards[counter - 1] = 0;

		if(counter == cards.length){
			shuffleCards();
			drawnCard = drawCard();
		}

		return drawnCard;
	}

	private int sumCards(){
		int handVal = 0;

		for(int card : hand){

			if(card > 10){
				handVal += 10;
			} else if(card == 1){

				if(handVal + 11 > 21){
					handVal += 1;
				} else{
					handVal += 11;
				}

			} else{
				handVal += card;
			}

		}

		return handVal;
	}

	public int hit(){
		
		if(!standing){
			hand[numCards] = drawCard();
			handVal = sumCards();
			numCards++;

			if(handVal >= 21){
				standing = true;
			}

			return hand[numCards - 1];
		} else{
			return -1;
		}

	}

	public boolean isStanding(){return standing;}
	public int getHand(){return handVal;}
	public SimpleBooleanProperty shuffledProperty(){return shuffled;}
}

class Generation{
	private static Random rand;
	private static int populationSize = 0;
	private static double elitePortion = 0;

	private Child[] children;

	public static void setRandom(Random rand){
		Generation.rand = rand;
	}

	public static void setParameters(int populationSize, double mutationChance, double mutationMagnitude, double maxWeightMagnitude, double crossoverChance, double elitePortion){
		Generation.populationSize = populationSize;
		Generation.elitePortion = elitePortion;
		Child.setParameters(mutationChance, mutationMagnitude, maxWeightMagnitude, crossoverChance);
	}

	public Generation(int[] architecture, int inputs){

		if(populationSize != 0){
			children = new Child[populationSize];

			for(int i = 0; i < populationSize; i++){
				children[i] = new Child(architecture, inputs);
			}

		} else{
			throw new RuntimeException("Genetic algorithm parameters not set!");
		}

	}

	public void generateNew(){
		Child[] newChildren = new Child[populationSize];
		organizeChildren();

		for(int i = 0; i < (int) (populationSize * elitePortion); i++){
			newChildren[i] = children[i];
			newChildren[i].setFitness(0);
		}

		for(int i = (int) (populationSize * elitePortion); i < populationSize; i++){
			newChildren[i] = new Child(selectChild(), selectChild());
		}

		children = newChildren;
	}

	private void organizeChildren(){
		Child temp;

		for(int i = 0; i < populationSize; i++){

			for(int j = i + 1; j < populationSize; j++){

				if(children[j].getFitness() > children[i].getFitness()){
					temp = children[i];
					children[i] = children[j];
					children[j] = temp;
				}

			}

		}

	}

	private void shuffleChildren(){
		Child temp;
		int newIndex;

		for(int i = 0; i < populationSize; i++){
			newIndex = rand.nextInt(populationSize);
			temp = children[0];
			children[0] = children[newIndex];
			children[newIndex] = temp;
		}

	}

	private Child selectChild(){
		double fitnessSum = 0;
		shuffleChildren();

		for(int i = 0; i < populationSize; i++){
			fitnessSum += children[i].getFitness();
		}

		double chosenSum = fitnessSum * rand.nextDouble();
		double currentSum = 0;

		for(int i = 0; i < populationSize; i++){
			currentSum += children[i].getFitness();

			if(currentSum > chosenSum){
				return children[i];
			}

		}

		return children[children.length - 1];
	}

	public Child getBest(){
		double bestFitness = 0;
		Child bestChild = null;

		for(Child child : children){

			if(child.getFitness() >= bestFitness){
				bestChild = child;
				bestFitness = child.getFitness();
			}

		}

		return bestChild;
	}

	public Child[] getChildren(){return children;}
}

class Child{
	private static Random rand;
	private static double mutationChance = 0;
	private static double mutationMagnitude = 0;
	private static double crossoverChance = 0;
	private static double maxWeightMagnitude = 0;

	private NeuralNetwork network;
	private double fitness = 0;

	public Child clone(){
		return new Child(network.getWeights(), network.getBiases(), network.getHiddenWeights());
	}

	public static void setRandom(Random rand){
		Child.rand = rand;
	}

	public static void setParameters(double mutationChance, double mutationMagnitude, double maxWeightMagnitude, double crossoverChance){
		Child.mutationChance = mutationChance;
		Child.mutationMagnitude = mutationMagnitude;
		Child.maxWeightMagnitude = maxWeightMagnitude;
		Child.crossoverChance = crossoverChance;
	}

	public Child(int[] architecture, int inputs){
		network = new NeuralNetwork(architecture, inputs);
	}

	public Child(NeuralNetwork network){
		this.network = network;
	}

	public Child(double[][][] weights, double[][] biases, double[][][] hiddenWeights){
		network = new NeuralNetwork(weights, biases, hiddenWeights);
	}

	private double[][][] crossWeights(double[][][] weights_1, double[][][] weights_2){
		double[][][] weights_3 = new double[weights_1.length][][];

		for(int i = 0; i < weights_1.length; i++){
			weights_3[i] = new double[weights_1[i].length][weights_1[i][0].length];

			for(int j = 0; j < weights_1[i].length; j++){

				for(int k = 0; k < weights_1[i][j].length; k++){

					if(rand.nextDouble() > 0.5){
						weights_3[i][j][k] = weights_1[i][j][k];
					} else{
						weights_3[i][j][k] = weights_2[i][j][k];
					}

				}

			}

		}

		return weights_3;
	}

	private double[][] crossBiases(double[][] biases_1, double[][] biases_2){
		double[][] biases_3 = new double[biases_1.length][];

		for(int i = 0; i < biases_1.length; i++){
			biases_3[i] = new double[biases_1[i].length];

			for(int j = 0; j < biases_1[i].length; j++){

				if(rand.nextDouble() > 0.5){
					biases_3[i][j] = biases_1[i][j];
				} else{
					biases_3[i][j] = biases_2[i][j];
				}

			}

		}

		return biases_3;
	}

	private void mutateWeights(double[][][] weights){

		for(int i = 0; i < weights.length; i++){

			for(int j = 0; j < weights[i].length; j++){

				for(int k = 0; k < weights[i][j].length; k++){

					if(rand.nextDouble() < mutationChance){
						weights[i][j][k] += rand.nextGaussian() * mutationMagnitude;
						
						if(Math.abs(weights[i][j][k]) > maxWeightMagnitude){
							weights[i][j][k] = maxWeightMagnitude * Math.signum(weights[i][j][k]);
						}

					}

				}

			}

		}

	}

	private void mutateBiases(double[][] biases){

		for(int i = 0; i < biases.length; i++){

			for(int j = 0; j < biases[i].length; j++){

				if(rand.nextDouble() < mutationChance){
					biases[i][j] += rand.nextGaussian() * mutationMagnitude;

					if(Math.abs(biases[i][j]) > maxWeightMagnitude){
						biases[i][j] = maxWeightMagnitude * Math.signum(biases[i][j]);
					}

				}

			}

		}

	}

	public Child(Child parent_1, Child parent_2){
		double[][][] weights_3;
		double[][] biases_3;
		double[][][] hiddenWeights_3;

		double[][][] weights_1 = parent_1.getNetwork().getWeights();
		double[][][] weights_2 = parent_2.getNetwork().getWeights();

		double[][] biases_1 = parent_1.getNetwork().getBiases();
		double[][] biases_2 = parent_2.getNetwork().getBiases();

		double[][][] hiddenWeights_1 = parent_1.getNetwork().getHiddenWeights();
		double[][][] hiddenWeights_2 = parent_2.getNetwork().getHiddenWeights();

		if(rand.nextDouble() < crossoverChance){
			weights_3 = crossWeights(weights_1, weights_2);
			biases_3 = crossBiases(biases_1, biases_2);
			hiddenWeights_3 = crossWeights(hiddenWeights_1, hiddenWeights_2);

			mutateWeights(weights_3);
			mutateBiases(biases_3);
			mutateWeights(hiddenWeights_3);
		} else{

			if(rand.nextDouble() < 0.5){
				weights_3 = weights_1;
				biases_3 = biases_1;
				hiddenWeights_3 = hiddenWeights_1;
			} else{
				weights_3 = weights_2;
				biases_3 = biases_2;
				hiddenWeights_3 = hiddenWeights_2;
			}

		}

		network = new NeuralNetwork(weights_3, biases_3, hiddenWeights_3);
	}

	public double[] recurrentPassThru(double[] inputs){
		return network.recurrentPassThru(inputs);
	}

	public void setFitness(double fitness){this.fitness = fitness;}
	public NeuralNetwork getNetwork(){return network;}
	public double getFitness(){return fitness;}
}

class NeuralNetwork{
	private static Random rand;
	private NeuralLayer[] layers;

	public static void setRandom(Random rand){
		NeuralNetwork.rand = rand;
	}

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

	public NeuralNetwork(NeuralLayer[] layers){
		this.layers = layers;

		if(!layersMatch()){
			throw new RuntimeException("Dimension mismatch between layers!");
		}

	}

	public NeuralNetwork(int[] architecture, int inputs){
		layers = new NeuralLayer[architecture.length];
		layers[0] = new NeuralLayer(inputs, architecture[0]);

		for(int i = 1; i < architecture.length; i++){
			layers[i] = new NeuralLayer(architecture[i - 1], architecture[i]);
		}

		if(!layersMatch()){
			throw new RuntimeException("Dimension mismatch between layers!");
		}

	}

	public NeuralNetwork(double[][][] weights, double[][] biases){

		if(weights.length != biases.length){
			throw new RuntimeException("Dimensions mismatch between weights and biases!");
		}

		layers = new NeuralLayer[weights.length];

		for(int i = 0; i < weights.length; i++){
			layers[i] = new NeuralLayer(weights[i], biases[i]);
		}

		if(!layersMatch()){
			throw new RuntimeException("Dimension mismatch between layers!");
		}

	}

	public NeuralNetwork(double[][][] weights, double[][] biases, double[][][] hiddenWeights){

		if(weights.length != biases.length){
			throw new RuntimeException("Dimensions mismatch between weights and biases!");
		}

		layers = new NeuralLayer[weights.length];

		for(int i = 0; i < weights.length; i++){
			layers[i] = new NeuralLayer(weights[i], biases[i], hiddenWeights[i]);
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

	public double[] recurrentPassThru(double[] inputs){
		double[] outputs = inputs;

		for(int i = 0; i < layers.length; i++){
			outputs = layers[i].recurrentPassThru(outputs);
		}

		return outputs;
	}

	public void resetHidden(){
		
		for(NeuralLayer layer : layers){
			layer.resetHidden();
		}

	}

	public double[][][] getWeights(){
		double[][][] weights = new double[layers.length][][];

		for(int i = 0; i < layers.length; i++){
			weights[i]  = layers[i].getWeights();
		}

		return weights;
	}

	public double[][] getBiases(){
		double[][] biases = new double[layers.length][];

		for(int i = 0; i < layers.length; i++){
			biases[i] = layers[i].getBiases();
		}

		return biases;
	}

	public double[][][] getHiddenWeights(){
		double[][][] hiddenWeights = new double[layers.length][][];

		for(int i = 0; i < layers.length; i++){
			hiddenWeights[i] = layers[i].getHiddenWeights();
		}

		return hiddenWeights;
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
	private double[][] hiddenWeights;
	private double[] hiddenInputs;

	public static void setRandom(Random rand){
		NeuralLayer.rand = rand;
	}

	public NeuralLayer(int inputs, int outputs){
		genWeights(inputs, outputs);
		genBiases(outputs);
		genHiddenWeights(inputs, outputs);
		this.hiddenInputs = new double[inputs];
	}

	public NeuralLayer(double[][] weights, double[] biases, double[][] hiddenWeights){
		this.weights = weights;
		this.biases = biases;
		this.hiddenWeights = hiddenWeights;
		this.hiddenInputs = new double[hiddenWeights.length];

		if(biases.length != weights.length){
			throw new RuntimeException("Dimension mismatch between biases and weights!");
		}

		if(hiddenWeights.length != weights[0].length || hiddenWeights[0].length != weights.length){
			throw new RuntimeException("Dimension mismatch between hidden weights and weights!");
		}

	}

	public NeuralLayer(double[][] weights, double[] biases){
		this.weights = weights;
		this.biases = biases;

		if(biases.length != weights.length){
			throw new RuntimeException("Dimension mismatch between biases and weights!");
		}

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

	public void genHiddenWeights(int inputs, int outputs){
		hiddenWeights = new double[inputs][outputs];

		for(int i = 0; i < inputs; i++){

			for(int j = 0; j < outputs; j++){
				hiddenWeights[i][j] = rand.nextGaussian();
			}

		}

	}

	private double ReLU(double input){
		
		if(input > 0){
			return input;
		} else{
			return 0;
		}

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
		}

		return outputs;
	}

	public double[] recurrentPassThru(double[] inputs){

		if(inputs.length != weights[0].length){
			throw new RuntimeException("Dimension mismatch between inputs and weights!");
		}

		if(hiddenWeights == null){
			throw new RuntimeException("Network layer is not recurrent!");
		}

		double[] outputs = new double[weights.length];

		for(int i = 0; i < weights.length; i++){

			for(int j = 0; j < inputs.length; j++){
				outputs[i] += weights[i][j] * (inputs[j] + hiddenInputs[j]);
				hiddenInputs[j] = 0;
			}

			outputs[i] += biases[i];
			outputs[i] = ReLU(outputs[i]);

			for(int j = 0; j < inputs.length; j++){
				hiddenInputs[j] += outputs[i] * hiddenWeights[j][i];
			}

		}

		return outputs;
	}

	public void resetHidden(){
		this.hiddenInputs = new double[hiddenWeights.length];
	}

	public double[][] getHiddenWeights(){return hiddenWeights;}
	public double[][] getWeights(){return weights;}
	public double[] getBiases(){return biases;}
}