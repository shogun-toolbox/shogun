using System;

using org.shogun;
using org.jblas;
public class kernel_io_modular
{
	static kernel_io_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Library");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double width = 1.2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_test, width);
		DoubleMatrix km_train = kernel.get_kernel_matrix();
		AsciiFile f =new AsciiFile("gaussian_train.ascii",'w');
		kernel.save(f);

		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		AsciiFile f_test =new AsciiFile("gaussian_train.ascii",'w');
		kernel.save(f_test);

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		Features.exit_shogun();
	}
}
