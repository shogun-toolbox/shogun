using System;

public class kernel_match_word_string_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int degree = 20;
		double scale = 1.4;
		int size_cache = 10;
		int order = 3;
		int gap = 0;
		bool reverse = false;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringWordFeatures feats_train = new StringWordFeatures(EAlphabet.DNA);
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		StringCharFeatures charfeat_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);
		StringWordFeatures feats_test = new StringWordFeatures(EAlphabet.DNA);
		feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);

		MatchWordStringKernel kernel = new MatchWordStringKernel(size_cache, degree);
		kernel.set_normalizer(new AvgDiagKernelNormalizer(scale));
		kernel.init(feats_train, feats_train);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();
	}
}