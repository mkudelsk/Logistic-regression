package logisticRegression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
 
public class DataSet {

	/** For storing data */
	protected double [][] X;
	protected int[] Y;
	
	/** Original variable names */
	protected String[] varNames;

	/** Constructor */
	public DataSet() {
	}

	/** Counts number of lines in the input file */
	private int countLines(String filename) throws IOException {
	    InputStream is = new BufferedInputStream(new FileInputStream(filename));
	    try {
	        byte[] c = new byte[1024];
	        int count = 0;
	        int readChars = 0;
	        boolean empty = true;
	        while ((readChars = is.read(c)) != -1) {
	            empty = false;
	            for (int i = 0; i < readChars; ++i) {
	                if (c[i] == '\n') {
	                    ++count;
	                }
	            }
	        }
	        return (count == 0 && !empty) ? 1 : count;
	    } finally {
	        is.close();
	    }
	}
	
	/** Read the input data from a .csv file 
	 * Assuming that:
	 * 	1) first line contains a header
	 *  2) first column contains the row number
	 *  3) second column contains the target variable
	 *  4) columns are separated with ","
	 * */
	public void readDataSet(String fileName) {
		BufferedReader fileReader = null;
        
        // Delimiter used in CSV file
        final String DELIMITER = ",";
        try
        {
    		// First, determine the number of lines in the file
    		int linesNumber = countLines(fileName);
    		
    		// allocate memory for rows
    		X = new double[linesNumber-1][];
    		Y = new int[linesNumber-1];
    		
            String line = "";
            // Create the file reader
            fileReader = new BufferedReader(new FileReader(fileName));
            
           // Read the variable names
           line = fileReader.readLine();
           varNames = line.split(DELIMITER);
            
            // Read the file line by line
            int rowNumber = 0;
            while ((line = fileReader.readLine()) != null)
            {
                // Get all columns available in line
                String[] columns = line.split(DELIMITER);
                
            	double[] x = new double[columns.length-1];
            	
            	// Add bias as x[0]
            	x[0] = 1.0;
            	
            	// Skip first column (it is the observation number)
            	// also: start from x[1]
                for(int i = 2; i < columns.length; i++) {
                	x[i-1] = Double.parseDouble(columns[i]);
                }
                
                // store values in memory
                X[rowNumber] = x;
                Y[rowNumber++] = Integer.parseInt(columns[1]);                
                
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        finally
        {
            try {
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
	}

	/** Write data to a .csv file */
	public void writeDataSet(String fileName) {
		BufferedWriter fileWriter = null;
		
		try
        {
			File file = new File(fileName);
			if (!file.exists())
				file.createNewFile();

			fileWriter = new BufferedWriter(new FileWriter(file));

			DecimalFormat df = new DecimalFormat("##.###");
			df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
			
			for(int i=0; i<varNames.length; i++) {
				fileWriter.write(varNames[i]);
				fileWriter.write(",");
			}
			fileWriter.write("Predicted " + varNames[1]);
			fileWriter.newLine();
			
			for(int i=0; i<X.length; i++) {
				fileWriter.write(Integer.toString(i+1));
				fileWriter.write(",");
				fileWriter.write(Integer.toString(Y[i]));
				fileWriter.write(",");
				for (int j=0; j<X[i].length; j++) {
					fileWriter.write(df.format(X[i][j]));
					
					if(j<X[i].length-1)
						fileWriter.write(",");
				}
				fileWriter.newLine();
			}
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        finally
        {
            try {
            	fileWriter.flush();
            	fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
	}
	
	/** Write data with the predictions appended at the end to a .csv file */
	public void writeDataSetPred(String fileName, double[] predictedY) {
		BufferedWriter fileWriter = null;
		
		try
        {
			File file = new File(fileName);
			if (!file.exists())
				file.createNewFile();

			fileWriter = new BufferedWriter(new FileWriter(file));
			
			DecimalFormat df = new DecimalFormat("##.###");
			df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
			
			for(int i=0; i<varNames.length; i++) {
				fileWriter.write(varNames[i]);
				fileWriter.write(",");
			}
			fileWriter.write("Predicted " + varNames[1]);
			fileWriter.newLine();
			
			for(int i=0; i<X.length; i++) {
				fileWriter.write(Integer.toString(i+1));
				fileWriter.write(",");
				fileWriter.write(Integer.toString(Y[i]));
				fileWriter.write(",");
				for (int j=0; j<X[i].length; j++) {
					fileWriter.write(df.format(X[i][j]));
					fileWriter.write(",");
				}
				fileWriter.write(df.format(predictedY[i]));
				fileWriter.newLine();
			}
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        finally
        {
            try {
            	fileWriter.flush();
            	fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
	}
	
	/** Print the input data set */
	public void printDataSet() {
		for(int i=0; i<X.length; i++) {
			System.out.print(Y[i] + "\t");
			for (int j=0; j<X[i].length; j++) {
				System.out.print(X[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	/** Convert predictor variables to a two dimensional array */
	public double[][] getX() {
		return X;
	}
	
	/** Convert predicted variable to an array */
	public int[] getY() {
		return Y;
	}
	
}


