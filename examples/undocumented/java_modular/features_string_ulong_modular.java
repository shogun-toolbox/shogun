import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_ulong_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public features_string_ulong_modular() {
		parameter_list.add(Arrays.asList(new Integer(0), new Integer(2), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(0), new Integer(3), new Integer(0)));
	}
	static ArrayList run(List para) {
		modshogun.init_shogun_with_defaults();
		boolean rev = false;
		int start = ((Integer)para.get(0)).intValue();
		int order = ((Integer)para.get(1)).intValue();
		int gap = ((Integer)para.get(2)).intValue();
		StringCharFeatures cf = new StringCharFeatures(new String[] { "hey", "guys", "string"}, RAWBYTE);
		StringUlongFeatures uf = new StringUlongFeatures(RAWBYTE);

		uf.obtain_from_char(cf, start,order,gap,rev);
		uf.set_feature_vector(new DoubleMatrix(new double[][] {{1,2,3,4,5}}), 0);

		ArrayList result = new ArrayList();
		result.add(uf.get_features());
		result.add(uf.get_feature_vector(2));
		result.add(uf.get_num_vectors());

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_ulong_modular x = new features_string_ulong_modular();
		run((List)x.parameter_list.get(0));
	}
}
