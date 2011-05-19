import shogun.*;
import org.jblas.*;
public class MatrixTest {
	static {
		System.loadLibrary("Features");
	}
	
	public static void main(String argv[]) {
		Features.init_shogun();
		System.out.println("Test DoubleMatrix(jblas):");
		RealFeatures x = new RealFeatures();		
		double y[][] = {{1, 2},{3, 4}, {5, 6}};
		DoubleMatrix A = new DoubleMatrix(y);
		x.set_feature_matrix(A);
		DoubleMatrix B = x.get_feature_matrix();
		for (int i = 0; i < 3; i ++) {
			for (int j = 0; j < 2; j++) {	
				System.out.println(B.get(i, j));
			}
		}
		Features.exit_shogun();
	}
}
