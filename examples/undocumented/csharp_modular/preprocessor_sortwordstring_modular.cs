using System;

public class preprocessor_sortwordstring_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		bool reverse = false;
		int order = 3;
		int gap = 0;

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommWordStringKernel kernel = new CommWordStringKernel(feats_train, feats_train, false);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}