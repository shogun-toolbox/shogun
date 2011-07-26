using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class kernel_exponential_modular
{
	static kernel_exponential_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_exponential_modular()
	{
		parameter_list.Add(Arrays.asList(new double?(1.0)));
		parameter_list.Add(Arrays.asList(new double?(5.0)));
	}
	public virtual object run(IList para)
	{
		modshogun.init_shogun_with_defaults();
		double tau_coef = (double)((double?)para[0]);

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);
		ExponentialKernel kernel = new ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10);

		kernel.init(feats_train, feats_train);
		DoubleMatrix km_train =kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test =kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.Add(km_train);
		result.Add(km_test);
		result.Add(kernel);

		modshogun.exit_shogun();
		return (object)result;
	}
	static void Main(string[] argv)
	{
		kernel_exponential_modular x = new kernel_exponential_modular();
		x.run((IList)x.parameter_list[0]);
	}
}
