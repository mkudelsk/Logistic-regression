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
		DenseMatrix64F HH_inv = new DenseMatrix64F(K,K);	// Inverted Hessian 
		
		LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.pseudoInverse(true);
		//LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.leastSquaresQrPivot(true,false);
		
		// Initialize y
		for (int i = 0; i < N; i++)
			y.set(i, 0, YY[i]);
	
		// Iterate until maxIterations or the STOP condition is met
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
			
			// Update weights according to the equation: beta_new = beta_old + ((Xt*W*X)^(-1))*Xt*(y-p)
			// First compute Hessian and try to invert it using pseudoInverse solver (can also invert a matrix close to singular)
			if(!solver.setA(Xt.mult(W).mult(X).getMatrix()) ) throw new RuntimeException("Invert failed");
	        solver.invert(HH_inv);
	        SimpleMatrix H_inv = new SimpleMatrix(HH_inv);

	        // Compute beta_new (weights)
	        beta_new.set( beta_old.plus(H_inv.mult(Xt).mult(y.minus(p))) );
			
			// Update weights based on beta_new
			for (int j=0; j<K; j++) {
				weights[j] = beta_new.get(j, 0);
			}
			
			// Compute log-likelihood
			for (int i=0; i<XX.length; i++) {
				double tmp = computePrediction(XX[i]);
				if (tmp == 0.0) 
					tmp = 0.000000001;
				else if (tmp == 1.0)
					tmp = 0.999999999;
				likelihood += YY[i] * Math.log(tmp) + (1-YY[i]) * Math.log(1- tmp);			
			}
			System.out.println(" Iteration " + n + ": log likelihood = " + likelihood);
	
			// Check STOP criteria
			maxWeightDev = 0.0;
			for (int j=0; j<weights.length; j++) {
				if((Math.abs(weights[j] - weightsPrev[j])/(Math.abs(weightsPrev[j]) + 0.01*minDelta)) > maxWeightDev) {
					maxWeightDev = (Math.abs(weights[j] - weightsPrev[j])/(Math.abs(weightsPrev[j]) + 0.01*minDelta));
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
	 * Use Stochastic Gradient Descent 
	 * */
	public void trainModelWithSGD(double[][] X, int[] Y) {
		double likelihood = 0.0;
		double[] weightsPrev = new double[weights.length];
		double maxWeightDev = 0.0;
		
		// Iterate until maxIterations or the STOP condition is met
		for (int n=0; n<maxIterations; n++) {
			likelihood = 0.0;
			
			// Store previous weights
			for (int j=0; j<weights.length; j++) weightsPrev[j] = weights[j];
			
			// Update weights
			for (int i=0; i<X.length; i++) {
				double yPredicted = computePrediction(X[i]);
							
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + learningRate * (Y[i] - yPredicted) * X[i][j];
				}
				// Compute log-likelihood
				likelihood += Y[i] * Math.log(computePrediction(X[i])) + (1-Y[i]) * Math.log(1- computePrediction(X[i]));
				
			}
			if (n%5000 == 0)
				System.out.println("Iteration " + n + ": log likelihood = " + likelihood);
			
			// Check STOP criteria
			maxWeightDev = 0.0;
			for (int j=0; j<weights.length; j++) {
				if((Math.abs(weights[j] - weightsPrev[j])/(Math.abs(weightsPrev[j]) + 0.01*minDelta)) > maxWeightDev) {
					maxWeightDev = (Math.abs(weights[j] - weightsPrev[j])/(Math.abs(weightsPrev[j]) + 0.01*minDelta));
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
	
	/** Set maxIterations */
	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}
	
	/** Set learning rate */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	
	/** Set minimum change of the weights (STOP criteria) */
	public void setMinDelta(double minDelta) {
		this.minDelta = minDelta;
	}
	
	/** Set classification cut off */
	public void setCutOff(double cutOff) {
		this.cutOff = cutOff;
	}
	
	public static void main(String[] args) {
		
		if (args.length != 2){
			System.out.println("Wrong command: there should be 2 parameters: path to the input file and training algorithm (SGD or IRLS)");
			return;
		}
		
        // Input file which needs to be parsed
        String inputFile = args[0];     
        
        // Output file: input file with the appended predictions
        String outputFile = args[0] + ".out";
        
        // Optimization parameters
        double learningRate = 0.0001; 	// Learning rate
        int maxIterations = 100000;		// Maximum number of iterations
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
        
        // Train model 
        switch(args[1]){
        	case("SGD"):
        		System.out.println("Training model with Stochastic Gradient Descent");
    			logistic.trainModelWithSGD(X, Y);
        		break;
        		
        	case("IRLS"):
        		logistic.setMaxIterations(15);	// this is usually enough for this method...
    			System.out.println("Training model with Newton-Raphson and Iterative Reweighted Least Squares");
    			logistic.trainModelWithIRLS(X, Y);
        	break;
        	
        	default:
        		System.out.println("Training model with Stochastic Gradient Descent");
        		logistic.trainModelWithSGD(X, Y);
        }
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