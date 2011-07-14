using System;

using org.shogun;
using org.jblas;
public class kernel_diag_modular
{
	static kernel_diag_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double diag = 23;

		DummyFeatures feats_train = new DummyFeatures(10);
		DummyFeatures feats_test = new DummyFeatures(17);

		ConstKernel kernel = new ConstKernel(feats_train, feats_train, diag);

		DoubleMatrix km_train = kernel.get_kernel_matrix();

		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test =kernel.get_kernel_matrix();

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		Features.exit_shogun();
	}
}
