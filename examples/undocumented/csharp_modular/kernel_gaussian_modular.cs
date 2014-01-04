using System;

public class kernel_gaussian_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 1.3;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		foreach(double item in km_train) {
			Console.Write(item);
		}

		foreach(double item in km_test) {
			Console.Write(item);
		}

	}
}
