using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.CUBE;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.BaumWelchViterbiType.BW_NORMAL;

public class distribution_hmm_modular
{
	static distribution_hmm_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_hmm_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(1), new int?(64), new double?(1e-5), new int?(3), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(1), new int?(64), new double?(1e-1), new int?(4), new int?(0)));
	}
	internal static ArrayList run(IList para)
	{
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		int N = (int)((int?)para[0]);
		int M = (int)((int?)para[1]);
		double pseudo = (double)((double?)para[2]);
		int order = (int)((int?)para[3]);
		int gap = (int)((int?)para[4]);

		string[] fm_train_dna = Load.load_cubes("../data/fm_train_cube.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, CUBE);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM hmm = new HMM(feats, N, M, pseudo);
		hmm.train();
		hmm.baum_welch_viterbi_train(BW_NORMAL);

		int num_examples = feats.get_num_vectors();
		int num_param = hmm.get_num_model_parameters();
		for (int i = 0; i < num_examples; i++)
		{
			for(int j = 0; j < num_param; j++)
			{
			hmm.get_log_derivative(j, i);
		}
		}

		int best_path = 0;
		int best_path_state = 0;
		for(int i = 0; i < num_examples; i++)
		{
			best_path += hmm.best_path(i);
			for(int j = 0; j < N; j++)
			{
				best_path_state += hmm.get_best_path_state(i, j);
			}
		}

		DoubleMatrix lik_example = hmm.get_log_likelihood();
		double lik_sample = hmm.get_log_likelihood_sample();

		ArrayList result = new ArrayList();
		result.Add(lik_example);
		result.Add(lik_sample);
		result.Add(hmm);
		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		distribution_hmm_modular x = new distribution_hmm_modular();
		run((IList)x.parameter_list[0]);
	}
}
