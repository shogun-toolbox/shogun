using System;

using org.shogun;
using org.jblas;
public class kernel_multiquadric_modular
{
	static kernel_multiquadric_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Distance");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double shift_coef = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		MultiquadricKernel kernel = new MultiquadricKernel(feats_train, feats_test, shift_coef, distance);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		Features.exit_shogun();
	}
}
