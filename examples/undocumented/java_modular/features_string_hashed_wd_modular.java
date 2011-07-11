import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWDNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class features_string_hashed_wd_modular {
	static {
		System.loadLibrary("Features");
	}

	public ArrayList parameter_list = new ArrayList();
	public features_string_hashed_wd_modular() {
		DoubleMatrix A = new DoubleMatrix(new double [][] {{1,2,3},{4,0,0},{0,0,0},{0,5,0},{0,0,6},{9,9,9}});
		parameter_list.add(Arrays.asList((A), new Integer(3), new Integer(1), new Integer(2)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		Features.init_shogun_with_defaults();
		DoubleMatrix A = (DoubleMatrix)para.get(0);
		int order = ((Integer)para.get(1)).intValue();
		int start_order = ((Integer)para.get(2)).intValue();
		int hash_bits = ((Integer)para.get(3)).intValue();
		
		int from_order = order;
		StringByteFeatures f = new StringByteFeatures(RAWDNA);
		//f.set_features(new DoubleMatrix(new double[][]{{0,1,2,3,0,1,2,3,3,2,2,1,1}}));
		
		HashedWDFeatures y = new HashedWDFeatures(f,start_order,order,from_order,hash_bits);
		//DoubleMatrix fm = y.get_feature_matrix();

		ArrayList result = new ArrayList();
		result.add(y);

		Features.exit_shogun();
		
		return result;
	}
	public static void main(String argv[]) {
		features_string_hashed_wd_modular x = new features_string_hashed_wd_modular();
		run((List)x.parameter_list.get(0));
	}
}
