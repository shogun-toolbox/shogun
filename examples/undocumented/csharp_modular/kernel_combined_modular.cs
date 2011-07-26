using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class kernel_combined_modular
{
	static kernel_combined_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_combined_modular()
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
		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		RealFeatures subfeats_train = new RealFeatures(traindata_real);
		RealFeatures subfeats_test = new RealFeatures(testdata_real);

		CombinedKernel kernel = new CombinedKernel();
		CombinedFeatures feats_train = new CombinedFeatures();
		CombinedFeatures feats_test = new CombinedFeatures();

		GaussianKernel subkernel = new GaussianKernel(10, 1.1);
		feats_train.append_feature_obj(subfeats_train);
		feats_test.append_feature_obj(subfeats_test);
		kernel.append_kernel(subkernel);

		StringCharFeatures subkfeats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures subkfeats_test = new StringCharFeatures(fm_test_dna, DNA);
		int degree = 3;
		FixedDegreeStringKernel subkernel2 = new FixedDegreeStringKernel(10, degree);
		feats_train.append_feature_obj(subkfeats_train);
		feats_test.append_feature_obj(subkfeats_test);
		kernel.append_kernel(subkernel2);

		subkfeats_train = new StringCharFeatures(fm_train_dna, DNA);
		subkfeats_test = new StringCharFeatures(fm_test_dna, DNA);
		LocalAlignmentStringKernel subkernel3 = new LocalAlignmentStringKernel(10);
		feats_train.append_feature_obj(subkfeats_train);
		feats_test.append_feature_obj(subkfeats_test);
		kernel.append_kernel(subkernel3);

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
		kernel_combined_modular x = new kernel_combined_modular();
		x.run((IList)x.parameter_list[0]);
	}
}
