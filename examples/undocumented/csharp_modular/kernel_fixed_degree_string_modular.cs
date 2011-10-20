using System;

public class kernel_fixed_degree_string_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int degree = 4;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);

		FixedDegreeStringKernel kernel = new FixedDegreeStringKernel(feats_train, feats_train, degree);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();
		modshogun.exit_shogun();
	}
}