using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class preprocessor_sortwordstring_modular
{
	static preprocessor_sortwordstring_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public preprocessor_sortwordstring_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(3), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(4), new int?(0)));
	}
	internal static ArrayList run(IList para)
	{
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = (int)((int?)para[0]);
		int gap = (int)((int?)para[1]);

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommWordStringKernel kernel = new CommWordStringKernel(feats_train, feats_train, false);

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
		preprocessor_sortwordstring_modular x = new preprocessor_sortwordstring_modular();
		run((IList)x.parameter_list[0]);
	}
}
