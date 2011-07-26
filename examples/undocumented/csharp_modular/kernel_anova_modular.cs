using System.Collections;

using org.shogun;
using org.jblas;

public class kernel_anova_modular
{
	static kernel_anova_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_anova_modular()
	{
		parameter_list.Add(Arrays.asList(new int?(2), new int?(10)));
		parameter_list.Add(Arrays.asList(new int?(5), new int?(10)));
	}
	public virtual object run(IList para)
	{
		modshogun.init_shogun_with_defaults();
		int cardinality = (int)((int?)para[0]);
		int size_cache = (int)((int?)para[1]);

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		ANOVAKernel kernel = new ANOVAKernel(feats_train, feats_train, cardinality, size_cache);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.Add(km_train);
		result.Add(km_test);
		result.Add(kernel);

		modshogun.exit_shogun();
		return (object)result;
	}
	static void Main(string[] argv)
	{
		kernel_anova_modular x = new kernel_anova_modular();
		x.run((IList)x.parameter_list[0]);
	}
}
