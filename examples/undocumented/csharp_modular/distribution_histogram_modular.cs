using System.Collections;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class distribution_histogram_modular
{
	static distribution_histogram_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Distribution");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_histogram_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(3), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(4), new int?(0)));
	}
	internal static ArrayList run(IList para)
	{
		bool reverse = false;
		Features.init_shogun_with_defaults();
		int order = (int)((int?)para[0]);
		int gap = (int)((int?)para[1]);

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		Histogram histo = new Histogram(feats);
		histo.train();

		histo.get_histogram();

		int num_examples = feats.get_num_vectors();
		int num_param = histo.get_num_model_parameters();

		DoubleMatrix out_likelihood = histo.get_log_likelihood();
		double out_sample = histo.get_log_likelihood_sample();

		ArrayList result = new ArrayList();
		result.Add(histo);
		result.Add(out_sample);
		result.Add(out_likelihood);
		Features.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		distribution_histogram_modular x = new distribution_histogram_modular();
		run((IList)x.parameter_list[0]);
	}
}
