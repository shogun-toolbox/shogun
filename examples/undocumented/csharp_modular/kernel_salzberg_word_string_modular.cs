using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class kernel_salzberg_word_string_modular
{
	static kernel_salzberg_word_string_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_salzberg_word_string_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(3), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(5), new int?(0)));
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
		feats_train.obtain_from_char(charfeat, order-1, order, gap, false);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, false);

		Labels labels = new Labels(Load.load_labels("../data/label_train_dna.dat"));

		PluginEstimate pie = new PluginEstimate();
		pie.set_labels(labels);
		pie.set_features(feats_train);
		pie.train();

		SalzbergWordStringKernel kernel = new SalzbergWordStringKernel(feats_train, feats_train, pie, labels);
		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		pie.set_features(feats_test);
		pie.apply().get_labels();
		DoubleMatrix km_test =kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.Add(km_train);
		result.Add(km_test);
		result.Add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		kernel_salzberg_word_string_modular x = new kernel_salzberg_word_string_modular();
		run((IList)x.parameter_list[0]);
	}
}
