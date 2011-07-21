import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_weighted_degree_position_string_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_weighted_degree_position_string_modular() {

		parameter_list.add(Arrays.asList(new Integer(3)));
		parameter_list.add(Arrays.asList(new Integer(20)));
	}
	static ArrayList run(List para) {
		modshogun.init_shogun_with_defaults();
		int degree = ((Integer)para.get(0)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		WeightedDegreePositionStringKernel kernel = new WeightedDegreePositionStringKernel(feats_train, feats_train, degree);
		kernel.set_position_weights(DoubleMatrix.ones(1, (fm_train_dna[0]).length()));
		

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		kernel_weighted_degree_position_string_modular x = new kernel_weighted_degree_position_string_modular();
		run((List)x.parameter_list.get(0));
	}
}
