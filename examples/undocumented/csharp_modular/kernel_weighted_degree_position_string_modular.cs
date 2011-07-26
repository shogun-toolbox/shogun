using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class kernel_weighted_degree_position_string_modular
{
	static kernel_weighted_degree_position_string_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_weighted_degree_position_string_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(3)));
		parameter_list.Add(Arrays.asList(new int?(20)));
	}
	internal static ArrayList run(IList para)
	{
		modshogun.init_shogun_with_defaults();
		int degree = (int)((int?)para[0]);

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		WeightedDegreePositionStringKernel kernel = new WeightedDegreePositionStringKernel(feats_train, feats_train, degree);
		kernel.set_position_weights(DoubleMatrix.ones(1, (fm_train_dna[0]).Length));


		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.Add(km_train);
		result.Add(km_test);
		result.Add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		kernel_weighted_degree_position_string_modular x = new kernel_weighted_degree_position_string_modular();
		run((IList)x.parameter_list[0]);
	}
}
