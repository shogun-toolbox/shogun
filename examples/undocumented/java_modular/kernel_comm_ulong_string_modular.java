import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;

public class kernel_comm_ulong_string_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 0;
		boolean reverse = false;
		boolean use_sign = false;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(DNA);
		charfeat.set_features(fm_train_dna);
		StringUlongFeatures feats_train = new StringUlongFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
		SortUlongString preproc = new SortUlongString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();

		StringCharFeatures charfeat_test = new StringCharFeatures(DNA);
		charfeat_test.set_features(fm_test_dna);
		StringUlongFeatures feats_test = new StringUlongFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommUlongStringKernel kernel = new CommUlongStringKernel(feats_train, feats_train, use_sign);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		modshogun.exit_shogun();
	}
}
