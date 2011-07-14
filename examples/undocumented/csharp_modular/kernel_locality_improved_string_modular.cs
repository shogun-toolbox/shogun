using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class kernel_locality_improved_string_modular
{
	static kernel_locality_improved_string_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		int length = 5;
		int inner_degree = 5;
		int outer_degree = 7;

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		LocalityImprovedStringKernel kernel = new LocalityImprovedStringKernel(feats_train, feats_train, length, inner_degree, outer_degree);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		Features.exit_shogun();
	}
}
