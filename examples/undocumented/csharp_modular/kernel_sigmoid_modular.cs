using System;

using org.shogun;
using org.jblas;
public class kernel_sigmoid_modular
{
	static kernel_sigmoid_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		int size_cache = 10;
		double gamma = 1.2;
		double coef0 = 1.3;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		SigmoidKernel kernel = new SigmoidKernel(feats_train, feats_test, size_cache, gamma, coef0);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		Features.exit_shogun();
	}
}
