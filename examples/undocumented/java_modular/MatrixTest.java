import org.shogun.*;
import org.jblas.*;

public class MatrixTest {
	static {
		System.loadLibrary("modshogun");
	}
	
	public static void main(String argv[]) {
		modshogun.init_shogun();
		System.out.println("Test DoubleMatrix(jblas):");
		RealFeatures x = new RealFeatures();		
		double y[][] = {{1, 2},{3, 4}, {5, 6}};
		DoubleMatrix A = new DoubleMatrix(y);
		x.set_feature_matrix(A);
		DoubleMatrix B = x.get_feature_matrix();
		System.out.println(B.toString());
		modshogun.exit_shogun();
	}
}
