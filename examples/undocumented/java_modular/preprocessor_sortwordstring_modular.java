import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class preprocessor_sortwordstring_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Preprocessor");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public preprocessor_sortwordstring_modular() {

		parameter_list.add(Arrays.asList(new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(4), new Integer(0)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		Features.init_shogun_with_defaults();
		int order = ((Integer)para.get(0)).intValue();
		int gap = ((Integer)para.get(1)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommWordStringKernel kernel = new CommWordStringKernel(feats_train, feats_train, false);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);
		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		preprocessor_sortwordstring_modular x = new preprocessor_sortwordstring_modular();
		run((List)x.parameter_list.get(0));
	}
}
