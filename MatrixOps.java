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

	public static double[] pointwiseMult(double[] vec1, double[] vec2){

		if(vec1.length != vec2.length){
			throw new RuntimeException("Dimension mismatch between vectors!");
		}

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] * vec2[i];
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

	public static double[] pointwiseAdd(double[] vec1, double[] vec2){

		if(vec1.length != vec2.length){
			throw new RuntimeException("Dimension mismatch between vectors!");
		}

		double[] newVec = new double[vec1.length];

		for(int i = 0; i < vec1.length; i++){
			newVec[i] = vec1[i] + vec2[i];
		}

		return newVec;
    }
    
    public static double[][] matrixMult(double[][] mat1, double[][] mat2){

        if(mat1[0].length != mat2.length){
            throw new RuntimeException("Dimension mismatch between matrices!");
        }

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

    public static double[] matrixMult(double[][] mat, double[] vec){

        if(mat[0].length != vec.length){
            throw new RuntimeException("Dimension mismatch between matrices!");
        }

        double[] newVec = new double[mat.length];

        for(int i = 0; i < mat.length; i++){

            for(int j = 0; j < vec.length; j++){
                newVec[i] += mat[i][j] * vec[j];
            }

        }

        return newVec;
    }

}