using System;

using org.shogun;
using org.jblas;
public class kernel_linear_modular
{
	static kernel_linear_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double scale = 1.2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		LinearKernel kernel = new LinearKernel(feats_train, feats_test);
		kernel.set_normalizer(new AvgDiagKernelNormalizer(scale));
		kernel.init(feats_train, feats_train);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		Features.exit_shogun();
	}
}
