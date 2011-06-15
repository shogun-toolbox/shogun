import org.shogun.*;
import org.jblas.*;
public class Load {
	public static DoubleMatrix load_numbers(String filename) {
		DoubleMatrix result = null;
		try {
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.reshape(2, temp.getLength() / 2);
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}	
	public static DoubleMatrix load_labels(String filename) {
		DoubleMatrix result = null;
		try {
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.reshape(1, temp.getLength());
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}
}	
