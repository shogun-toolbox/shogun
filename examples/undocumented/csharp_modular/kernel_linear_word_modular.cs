using System;

using org.shogun;
using org.jblas;

public class kernel_linear_word_modular
{
	static kernel_linear_word_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();
		double scale = 1.2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_word.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_word.dat");

		WordFeatures feats_train = new WordFeatures(traindata_real);
		WordFeatures feats_test = new WordFeatures(testdata_real);

		LinearKernel kernel = new LinearKernel(feats_train, feats_test);
		kernel.set_normalizer(new AvgDiagKernelNormalizer(scale));
		kernel.init(feats_train, feats_train);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		Console.WriteLine(km_train.ToString());
		Console.WriteLine(km_test.ToString());

		modshogun.exit_shogun();
	}
}
