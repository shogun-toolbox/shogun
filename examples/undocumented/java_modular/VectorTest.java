import org.shogun.*;
import org.jblas.*;
public class VectorTest {
	static {
		System.loadLibrary("Features");
	}

	public static void main(String argv[]) {
		Features.init_shogun();
		Labels x = new Labels();

		double y[][] = {{1, 2, 3, 4}};
		DoubleMatrix A = new DoubleMatrix(y);
		x.set_labels(A);
		DoubleMatrix B = x.get_labels();
		System.out.println(B.toString());
		Features.exit_shogun();
	}
}
