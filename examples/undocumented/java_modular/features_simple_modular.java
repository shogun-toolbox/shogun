import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_simple_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList();
	public features_simple_modular() {
		DoubleMatrix A = new DoubleMatrix(new double [][] {{1,2,3},{4,0,0},{0,0,0},{0,5,0},{0,0,6},{9,9,9}});
		parameter_list.add(Arrays.asList(A));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		DoubleMatrix A = (DoubleMatrix)para.get(0);

		RealFeatures a = new RealFeatures(A);		
		LongIntFeatures b = new LongIntFeatures(A);
		ByteFeatures c = new ByteFeatures(A);
		a.set_feature_vector(new DoubleMatrix(new double[][] {{1,4,0,0,0,9}}), 0);
		DoubleMatrix a_out = a.get_feature_matrix();
		DoubleMatrix b_out = b.get_feature_matrix();
		DoubleMatrix c_out = c.get_feature_matrix();

		ArrayList result = new ArrayList();
		result.add(a_out);
		result.add(b_out);
		result.add(c_out);
		result.add(a);
		result.add(a);
		result.add(c);
		modshogun.exit_shogun();
		
		return result;
	}
	public static void main(String argv[]) {
		features_simple_modular x = new features_simple_modular();
		run((List)x.parameter_list.get(0));
	}
}
