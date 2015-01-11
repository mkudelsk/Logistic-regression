package logisticRegression;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.simple.SimpleMatrix;

public class LogisticRegression {
	/** Weights to learn */
	protected double[] weights;
	
	/** Learning rate */
	protected double learningRate;
	
	/** Maximum number of iterations */
	protected int maxIterations;
	
	/** Minimum change of the weights (STOP criteria) */
	protected double minDelta;
	
	/** Classification cut off */
	protected double cutOff;
	
	/** Constructor */
	public LogisticRegression(int n) {
		weights = new double[n];
		this.learningRate = 0.0001;
		this.maxIterations = 3000;
		this.minDelta = 0.0001;
		this.cutOff = 0.5;
	}
	
	/** Constructor */
	public LogisticRegression(int n, double learningRate, int maxIterations, double minDelta, double cutOff) {
		weights = new double[n];
		this.learningRate = learningRate;
		this.maxIterations = maxIterations;
		this.minDelta = minDelta;
		this.cutOff = cutOff;
	}
	
	/** Sigmoid function */
	protected double sigmoid(double z) {
		return (1.0 / (1.0 + Math.exp(-z)));
	}
	
	/** Training function 
	 * Use Newton-Raphson and Iterative Reweighted Least Squares 
	 * */
	public void trainModelWithIRLS(double[][] XX, int[] YY) {
		int N = YY.length;
		int K = weights.length;
		double likelihood = 0.0;
		double[] weightsPrev = new double[K];
		double maxWeightDev = 0.0;
		
		// Initialize matrices
		SimpleMatrix X = new SimpleMatrix(XX);				// Input data
		SimpleMatrix Xt = X.transpose();					// Input data (transposed) 
		SimpleMatrix W = new SimpleMatrix(N,N);				// Weighting matrix (diagonal)
		SimpleMatrix y = new SimpleMatrix(N,1);				// Labels (predicted variable)
		SimpleMatrix p = new SimpleMatrix(N,1);				// Predictions
		SimpleMatrix beta_new = new SimpleMatrix(K,1);		// Weights
		SimpleMatrix beta_old = new SimpleMatrix(K,1);		// Weights from the previous iteration
		
		LinearSolver<DenseMatrix64F> solver;
		
		// Initialize y
		for (int i = 0; i < N; i++)
			y.set(i, 0, YY[i]);
	
		
		//////////////////////////////////////////////////
		//
		// DEBUG ONLY !!!
		//
		//////////////////////////////////////////////////
		X.printDimensions();
		Xt.printDimensions();
		W.printDimensions();
		y.printDimensions();
		beta_new.printDimensions();
		beta_old.printDimensions();
		
		//SimpleMatrix TMP = X.extractMatrix(0, 17, 0, 17);
		//TMP.print();
		//TMP.invert().print();
		
		SimpleMatrix a = new SimpleMatrix(3,3);
		a.set(0,0,1);
		a.set(0,1,2);
		a.set(0,2,3);
		a.set(1,0,4);
		a.set(1,1,5);
		a.set(1,2,6);
		a.set(2,0,7);
		a.set(2,1,8);
		a.set(2,2,9);
		
		a.print();
		a.invert().print();
		
//		DenseMatrix64F aa = new DenseMatrix64F(3,3);
//		solver = LinearSolverFactory.leastSquaresQrPivot(true,false);
//		if( !solver.setA(a.getMatrix()) ) throw new RuntimeException("Invert failed");
//        solver.invert(aa);
//        aa.print();
		
		//////////////////////////////////////////////////
		//////////////////////////////////////////////////
		
		for (int n=0; n<maxIterations; n++) {
			likelihood = 0.0;
			
			// Store previous weights
			for (int j=0; j<K; j++) 
				weightsPrev[j] = weights[j];
			beta_old.set(beta_new);
			
			// Update W and p
			for (int i=0; i < N; i++) {
				double yPredicted = computePrediction(XX[i]);
				p.set(i, 0, yPredicted);
				W.set(i, i, yPredicted*(1-yPredicted) );
			}
			
/*			X.print();
			Xt.print();
			y.print();
			beta_new.print();
			beta_old.print();
			p.print();*/
			
			// Update weights according to the equation: beta_new = beta_old - ((Xt*W*X)^(-1))*Xt*(y-p)
			//Xt.mult(W).mult(X).invert().print();
			//beta_new = beta_old.minus(Xt.mult(W).mult(X).invert().mult(Xt).mult(y.minus(p)));
			//Xt.mult(W).print();
			//W.print();
			
			// SimpleMatrix H = Xt.mult(W).mult(X);
			//H.print();
			DenseMatrix64F HH_inv = new DenseMatrix64F(K,K);
			
			solver = LinearSolverFactory.pseudoInverse(true);
			//solver = LinearSolverFactory.leastSquaresQrPivot(true,false);
			if( !solver.setA(Xt.mult(W).mult(X).getMatrix()) ) throw new RuntimeException("Invert failed");
	        solver.invert(HH_inv);
	        //HH_inv.print();
	        SimpleMatrix H_inv = new SimpleMatrix(HH_inv);

	        beta_new.set( beta_old.plus(H_inv.mult(Xt).mult(y.minus(p))) );
	        //beta_new.transpose().print();
	        //y.minus(p).print();
			
			// Update weights based on beta_new
			for (int j=0; j<K; j++) {
				weights[j] = beta_new.get(j, 0);
				System.out.print(weights[j] + " , ");
			}
			System.out.println();
						
			for (int i=0; i<XX.length; i++) {
				likelihood += YY[i] * Math.log(computePrediction(XX[i])) + (1-YY[i]) * Math.log(1- computePrediction(XX[i]));			
			}
			System.out.println("Iteration " + n + ": log likelihood = " + likelihood);
	
			maxWeightDev = 0.0;
			for (int j=0; j<weights.length; j++) {
				if(Math.abs(weights[j] - weightsPrev[j]) > maxWeightDev) {
					maxWeightDev = Math.abs(weights[j] - weightsPrev[j]);
				}
			}
			if(maxWeightDev < minDelta) {
				System.out.println("STOP criteria met: Iteration " + n + ", Log-likelihood = " + likelihood);
				break;
			}
		}	
		System.out.println("Final log-likelihood= " + likelihood);
		System.out.println();
	}
	
	/** Training function 
	 * Use stochastic gradient descent 
	 * */
	public void trainModelWithSGD(double[][] X, int[] Y) {
		double likelihood = 0.0;
		double[] weightsPrev = new double[weights.length];
		double maxWeightDev = 0.0;
		
		for (int n=0; n<maxIterations; n++) {
			if ((n%5000) == 0)
				System.out.println(" iteration " + n + "...");
			likelihood = 0.0;
			for (int j=0; j<weights.length; j++) weightsPrev[j] = weights[j];
			
			for (int i=0; i<X.length; i++) {
				double yPredicted = computePrediction(X[i]);
							
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + learningRate * (Y[i] - yPredicted) * X[i][j];
				}
				// Compute log likelihood (purpose: only debugging and monitoring progress)
				likelihood += Y[i] * Math.log(computePrediction(X[i])) + (1-Y[i]) * Math.log(1- computePrediction(X[i]));
				
			}
			//System.out.println("iteration " + n + ": log likelihood = " + likelihood);
			
			maxWeightDev = 0.0;
			for (int j=0; j<weights.length; j++) {
				if(Math.abs(weights[j] - weightsPrev[j]) > maxWeightDev) {
					maxWeightDev = Math.abs(weights[j] - weightsPrev[j]);
				}
			}
			if(maxWeightDev < minDelta) {
				System.out.println("STOP criteria met: Iteration " + n + ", Log-likelihood = " + likelihood);
				break;
			}
		}	
		System.out.println("Final log-likelihood= " + likelihood);
		System.out.println();
	}	
	
	/** Use the model to compute prediction for the given x */
	private double computePrediction(double[] x) {
		double logit = 0.0;
		for (int i=0; i<weights.length; i++) {
			logit += weights[i] * x[i];
		}
		return sigmoid(logit);
	}

	/** Print the model */
	private void printModel() {
		DecimalFormat df = new DecimalFormat("###.###");
		df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
		System.out.println("Logistic regression weights:");
		for(int i=0; i<weights.length; i++) {
			System.out.print(df.format(weights[i]) + " ");
		}
		System.out.println();
	}
	
	/** Score data with the model */
	private double[] scoreData(double[][] data) {
		int n = data.length;
		double[] predictedY = new double[n];
		
		for(int i=0; i<n; i++) {
			predictedY[i] = computePrediction(data[i]);
		}
		return predictedY;
	}
	
	/** Compute error rates */
	private void computeErrors(int[] Y, double[] predictedY) {
		int FP = 0;
		int FN = 0;
		int TP = 0;
		int TN = 0;
		double FNR = 0.0;
		double FPR = 0.0;
		
		for(int i=0; i<predictedY.length; i++) {
			int predY = ((predictedY[i] >= cutOff) ? 1 : 0);
			if((Y[i] == 1) && (predY == 1)) TP += 1;
			else if((Y[i] == 0) && (predY == 1)) FP += 1;
			else if((Y[i] == 0) && (predY == 0)) TN += 1;
			else if((Y[i] == 1) && (predY == 0)) FN += 1;
		}
		
		FNR = 1.0 * FN / (TP + FN);
		FPR = 1.0 * FP / (TN + FP);
		
		DecimalFormat df = new DecimalFormat("##.###");
		df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
		
		System.out.println();
		System.out.println("Error rates (cutOff="+cutOff+"):");
		System.out.println("False Negative Rate= " + df.format(FNR));
		System.out.println("False Positive Rate= " + df.format(FPR));
		System.out.println("TP= " + TP + " FP= " + FP + " TN= " + TN + " FN= " + FN);
		System.out.println();
		
	}
	
	private double arrayMax(double[] dataArray) {
		double maxVal = 0.0;
		for(int i=0; i< dataArray.length; i++) {
			if(dataArray[i] > maxVal) maxVal = dataArray[i];
		}
		return maxVal;
	}
	
	private double arrayMin(double[] dataArray) {
		double minVal = Double.MAX_VALUE;
		for(int i=0; i< dataArray.length; i++) {
			if(dataArray[i] < minVal) minVal = dataArray[i];
		}
		return minVal;
	}
	
	/** Simple standardization */
	private void standardize(double[][] dataArray) {
		double[] varX = new double[dataArray.length];
		for(int i=0; i<dataArray[0].length; i++) {
			for(int j=0; j<dataArray.length; j++) {
				varX[j] = dataArray[j][i];
			}
			double maxVal = arrayMax(varX);
			double minVal = arrayMin(varX);
			if (maxVal == minVal)
				continue;
			for(int j=0; j<dataArray.length; j++) {
				dataArray[j][i] = (dataArray[j][i] - minVal)/(maxVal - minVal);
			}
		}
	}
	
	public static void main(String[] args) {
		
		if (args.length != 1){
			System.out.println("Wrong command: there should be one and only one parameter: path to the input file.");
			return;
		}
		
        // Input file which needs to be parsed
        String inputFile = args[0];     
        
        // Output file: input file with the appended predictions
        String outputFile = args[0] + ".out";
        
        // Optimization parameters
        double learningRate = 0.0001; 	// Learning rate
        int maxIterations = 10000;		// Maximum number of iterations
        double minDelta = 0.0001;		// Minimum change of the weights (STOP criteria)
    	double cutOff = 0.5;			// Classification cut off 
        
    	// Read input data from csv file
    	System.out.print("Loading data...");
        DataSet inputData = new DataSet();
        inputData.readDataSet(inputFile);
        System.out.println(" DONE.");
        
        // Print input data
        //inputData.printDataSet();
        
    	// Predictor variables 
    	double[][] X = inputData.getX();
    	
    	// Predicted variable 
    	int[] Y = inputData.getY();	    	 	  	
    	
    	// Create instance of the logistic regression 
        LogisticRegression logistic = new LogisticRegression(X[0].length, learningRate, maxIterations, minDelta, cutOff);
        
        // Standardize predictor variables 
        System.out.print("Scaling data...");
        logistic.standardize(X);
        System.out.println(" DONE.");
        
        // Train model with Stochastic Gradient Descent 
        System.out.println("Training model...");
        //logistic.trainModelWithSGD(X, Y);
        logistic.trainModelWithIRLS(X, Y);
        System.out.println("Training DONE.");
        System.out.println();
        
        // Print model 
        logistic.printModel();
        
        // Print model 
        double[] predictedY = logistic.scoreData(X);
        
        // Compute errors 
        logistic.computeErrors(Y, predictedY);
        
        // Save data with the appended predictions
        inputData.writeDataSetPred(outputFile, predictedY);
        
	}        
}