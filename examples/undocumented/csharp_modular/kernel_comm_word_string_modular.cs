using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class kernel_comm_word_string_modular
{
	static kernel_comm_word_string_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Preprocessor");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		int order = 3;
		int gap = 0;
		bool reverse = false;
		bool use_sign = false;

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(DNA);
		charfeat.set_features(fm_train_dna);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();

		StringCharFeatures charfeat_test = new StringCharFeatures(DNA);
		charfeat_test.set_features(fm_test_dna);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommWordStringKernel kernel = new CommWordStringKernel(feats_train, feats_train, use_sign);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		Features.exit_shogun();
	}
}
