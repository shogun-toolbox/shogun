using System;

public class kernel_locality_improved_string_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int length = 5;
		int inner_degree = 5;
		int outer_degree = 7;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);

		LocalityImprovedStringKernel kernel = new LocalityImprovedStringKernel(feats_train, feats_train, length, inner_degree, outer_degree);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();
	}
}
