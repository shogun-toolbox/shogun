using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class distribution_linearhmm_modular
{
	static distribution_linearhmm_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_linearhmm_modular()
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

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		LinearHMM hmm = new LinearHMM(feats);
		hmm.train();

		hmm.get_transition_probs();

		int num_examples = feats.get_num_vectors();
		int num_param = hmm.get_num_model_parameters();
		for (int i = 0; i < num_examples; i++)
		{
			for(int j = 0; j < num_param; j++)
			{
			hmm.get_log_derivative(j, i);
		}
		}

		DoubleMatrix out_likelihood = hmm.get_log_likelihood();
		double out_sample = hmm.get_log_likelihood_sample();

		ArrayList result = new ArrayList();
		result.Add(hmm);
		result.Add(out_sample);
		result.Add(out_likelihood);
		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		distribution_linearhmm_modular x = new distribution_linearhmm_modular();
		run((IList)x.parameter_list[0]);
	}
}
