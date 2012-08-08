using System;

public class kernel_log_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double degree = 2.0;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		WaveKernel kernel = new WaveKernel(feats_train, feats_test, degree, distance);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		foreach (double item in km_train)
		      Console.Write(item);

		foreach (double item in km_test)
		      Console.Write(item);

		modshogun.exit_shogun();
	}
}
