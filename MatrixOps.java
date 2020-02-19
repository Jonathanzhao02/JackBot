public class MatrixOps{

    public static double[] concatenate(double[] vec1, double[] vec2){
		double[] newVec = new double[vec1.length + vec2.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i];
		}

		for(int i = vec1.length; i < vec1.length + vec2.length; i++){
			newVec[i] = vec2[i - vec1.length];
		}

		return newVec;
	}

	public static double[] pointwiseReLU(double[] vec){
		double[] newVec = new double[vec.length];

		for(int i = 0; i < vec.length; i++){
			newVec[i] = Math.max(0, vec[i]);
		}

		return newVec;
	}

	public static double[] pointwiseLeakyReLU(double[] vec){
		double[] newVec = new double[vec.length];

		for(int i = 0; i < vec.length; i++){
			
			if(vec[i] > 0){
				newVec[i] = vec[i];
			} else{
				newVec[i] = GenericNetwork.LEAKY_SLOPE * vec[i];
			}

		}

		return newVec;
	}

	public static double[] pointwiseSigmoid(double[] vec){
		double[] newVec = new double[vec.length];

		for(int i = 0; i < vec.length; i++){
			newVec[i] = 1.0 / (1.0 + Math.pow(Math.E, -vec[i]));
		}

		return newVec;
	}

	public static double[] pointwiseTanh(double[] vec){
		double[] newVec = new double[vec.length];

		for(int i = 0; i < vec.length; i++){
			newVec[i] = Math.tanh(vec[i]);
		}

		return newVec;
	}

	public static double[] pointwiseSoftmax(double[] vec){
		double[] newVec = new double[vec.length];
		double sum = 0;

		for(int i = 0; i < vec.length; i++){
			sum += Math.pow(Math.E, vec[i]);
		}

		for(int i = 0; i < vec.length; i++){
			newVec[i] = Math.pow(Math.E, vec[i]) / sum;
		}

		return newVec;
	}

	public static double[] pointwiseFunction(double[] vec, ActivationFunction activationFunction){

		switch(activationFunction){
			case RELU:
				return pointwiseReLU(vec);
			case LEAKY_RELU:
				return pointwiseLeakyReLU(vec);
			case SIGMOID:
				return pointwiseSigmoid(vec);
			case TANH:
				return pointwiseTanh(vec);
			case SOFTMAX:
				return pointwiseSoftmax(vec);
			default:
				throw new RuntimeException("Unsupported activation function " + activationFunction.name());
		}

	}

	public static double[] pointwiseAdd(double[] vec1, double[] vec2){
		checkDimMatch(vec1, vec2);

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] + vec2[i];
		}

		return newVec;
	}

	public static double[][] pointwiseAdd(double[][] mat1, double[][] mat2){
		checkDimMatch(mat1, mat2);

		double[][] newMat = new double[mat1.length][mat1[0].length];

		for(int i = 0; i < mat1.length; i++){
			newMat[i] = pointwiseAdd(mat1[i], mat2[i]);
		}

		return newMat;
	}

	public static double[][][] pointwiseAdd(double[][][] ten1, double[][][] ten2){
		checkDimMatch(ten1, ten2);

		double[][][] newTen = new double[ten1.length][ten1[0].length][ten1[0][0].length];

		for(int i = 0; i < ten1.length; i++){
			newTen[i] = pointwiseAdd(ten1[i], ten2[i]);
		}

		return newTen;
	}

	public static double[] pointwiseSubtract(double[] vec1, double[] vec2){
		checkDimMatch(vec1, vec2);

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] - vec2[i];
		}

		return newVec;
	}

	public static double[][] pointwiseSubtract(double[][] mat1, double[][] mat2){
		checkDimMatch(mat1, mat2);

		double[][] newMat = new double[mat1.length][mat1[0].length];

		for(int i = 0; i < mat1.length; i++){
			newMat[i] = pointwiseSubtract(mat1[i], mat2[i]);
		}

		return newMat;
	}

	public static double[][][] pointwiseSubtract(double[][][] ten1, double[][][] ten2){
		checkDimMatch(ten1, ten2);

		double[][][] newTen = new double[ten1.length][ten1[0].length][ten1[0][0].length];

		for(int i = 0; i < ten1.length; i++){
			newTen[i] = pointwiseSubtract(ten1[i], ten2[i]);
		}

		return newTen;
	}

	public static double[] pointwiseMult(double[] vec1, double[] vec2){
		checkDimMatch(vec1, vec2);

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] * vec2[i];
		}

		return newVec;
	}
	
	public static double[][] pointwiseMult(double[][] mat1, double[][] mat2){
		checkDimMatch(mat1, mat2);

		double[][] newMat = new double[mat1.length][mat1[0].length];

		for(int i = 0; i < mat1.length; i++){
			newMat[i] = pointwiseMult(mat1[i], mat2[i]);
		}

		return newMat;
    }
    
    public static double[][] matrixMult(double[][] mat1, double[][] mat2){
		checkDimValidity(mat1, mat2);

        double[][] newMat = new double[mat1.length][mat2[0].length];

        for(int i = 0; i < mat1.length; i++){

            for(int j = 0; j < mat2[0].length; j++){

                for(int k = 0; k < mat2.length; k++){
                    newMat[i][j] += mat1[i][k] * mat2[k][j];
                }

            }

        }

        return newMat;
    }

	public static double[][] matrixMult(double[] vec1, double[] vec2){
		double[][] newMat = new double[vec1.length][vec2.length];

		for(int i = 0; i < vec1.length; i++){

			for(int j = 0; j < vec2.length; j++){
				newMat[i][j] = vec1[i] * vec2[j];
			}

		}

		return newMat;
	}

	public static double[] matrixMult(double[][] mat, double[] vec){
		checkDimValidity(mat, vec);

        double[] newVec = new double[mat.length];

        for(int i = 0; i < mat.length; i++){

            for(int j = 0; j < vec.length; j++){
                newVec[i] += mat[i][j] * vec[j];
            }

        }

        return newVec;
	}

	public static double[] scalarMult(double[] vec, double val){
		double[] newVec = vec.clone();

		for(int i = 0; i < vec.length; i++){
			newVec[i] *= val;
		}

		return newVec;
	}

	public static double[][] scalarMult(double[][] mat, double val){
		double[][] newMat = mat.clone();

		for(int i = 0; i < mat.length; i++){
			newMat[i] = scalarMult(mat[i], val);
		}

		return newMat;
	}

	public static double[][][] scalarMult(double[][][] ten, double val){
		double[][][] newTen = ten.clone();

		for(int i = 0; i < ten.length; i++){
			newTen[i] = scalarMult(ten[i], val);
		}

		return newTen;
	}
	
	public static double[][] transpose(double[][] mat){
		double[][] newMat = new double[mat[0].length][mat.length];

		for(int i = 0; i < mat[0].length; i++){

			for(int j = 0; j < mat.length; j++){
				newMat[i][j] = mat[j][i];
			}

		}

		return newMat;
	}

	public static double[][] transpose(double[] vec){
		double[][] newMat = new double[0][vec.length];

		for(int i = 0; i < vec.length; i++){
			newMat[0][i] = vec[i];
		}

		return newMat;
	}

	public static void checkDimMatch(double[] vec1, double[] vec2){

		if(vec1.length != vec2.length){
			throw new RuntimeException("Dimension mismatch between vectors! (" + vec1.length + " and " + vec2.length + ")");
		}

	}

	public static void checkDimMatch(double[][] mat1, double[][] mat2){

		if(mat1.length != mat2.length || mat1[0].length != mat2[0].length){
			throw new RuntimeException("Dimension mismatch between matrices! (" + mat1.length + "x" + mat1[0].length + " and " + mat2.length + "x" + mat2[0].length + ")");
		}

	}

	public static void checkDimMatch(double[][][] ten1, double[][][] ten2){

		if(ten1.length != ten2.length || ten1[0].length != ten2[0].length || ten1[0][0].length != ten2[0][0].length){
			throw new RuntimeException("Dimension mismatch between tensors! (" + ten1.length + "x" + ten1[0].length + "x" + ten1[0][0].length + " and " + ten2.length + "x" + ten2[0].length + "x" + ten2[0][0].length + ")");
		}

	}

	public static void checkDimValidity(double[][] mat, double[] vec){

		if(mat[0].length != vec.length){
			throw new RuntimeException("Dimension mismatch between matrices! (" + mat.length + "x" + mat[0].length + " and " + vec.length + "x1)");
		}

	}

	public static void checkDimValidity(double[][] mat1, double[][] mat2){

		if(mat1[0].length != mat2.length){
			throw new RuntimeException("Dimension mismatch between matrices! (" + mat1.length + "x" + mat1[0].length + " and " + mat2.length + "x" + mat2[0].length + ")");
		}

	}

}