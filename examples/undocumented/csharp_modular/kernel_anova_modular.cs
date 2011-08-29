using System;

public class kernel_anova_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int cardinality = 2;
		int size_cache = 5;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		ANOVAKernel kernel = new ANOVAKernel(feats_train, feats_train, cardinality, size_cache);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
