using System;

public class kernel_poly_match_word_string_modular {
	public static void Main() {

		modshogun.init_shogun_with_defaults();

		bool reverse = false;
		int order = 3;
		int gap = 0;
		int degree = 2;

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, false);

		charfeat = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, false);

		PolyMatchWordStringKernel kernel = new PolyMatchWordStringKernel(feats_train, feats_train, degree, true);
		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test=kernel.get_kernel_matrix();

	}
}