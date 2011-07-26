using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.BaumWelchViterbiType.BW_NORMAL;

public class kernel_fisher_modular
{
	static kernel_fisher_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_fisher_modular()
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

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");
		DoubleMatrix label_train_dna = Load.load_labels("../data/label_train_dna.dat");


		ArrayList fm_hmm_pos_builder = new ArrayList();
		ArrayList fm_hmm_neg_builder = new ArrayList();
		for(int i = 0; i < label_train_dna.Columns; i++)
		{
			if (label_train_dna.get(i) == 1)
			{
				fm_hmm_pos_builder.Add(fm_train_dna[i]);
			}
			else
			{
				fm_hmm_neg_builder.Add(fm_train_dna[i]);
			}
		}

		int pos_size = fm_hmm_pos_builder.Count;
		int neg_size = fm_hmm_neg_builder.Count;
		string[] fm_hmm_pos = new string[pos_size];
		string[] fm_hmm_neg = new string[neg_size];
		for (int i = 0; i < pos_size; i++)
		{
			fm_hmm_pos[i] = (string)fm_hmm_pos_builder[i];
		}
		for (int i = 0; i < neg_size; i++)
		{
			fm_hmm_pos[i] = (string)fm_hmm_neg_builder[i];
		}

		StringCharFeatures charfeat = new StringCharFeatures(fm_hmm_pos, DNA);
		StringWordFeatures hmm_pos_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_pos_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM pos = new HMM(hmm_pos_train, N, M, pseudo);
		pos.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_hmm_neg, DNA);
		StringWordFeatures hmm_neg_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_neg_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM neg = new HMM(hmm_neg_train, N, M, pseudo);
		neg.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures wordfeats_train = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures wordfeats_test = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		pos.set_observations(wordfeats_train);
		neg.set_observations(wordfeats_train);
		FKFeatures feats_train = new FKFeatures(10, pos, neg);
		feats_train.set_opt_a(-1);
		PolyKernel kernel = new PolyKernel(feats_train, feats_train, 1, true);
		DoubleMatrix km_train = kernel.get_kernel_matrix();

		HMM pos_clone = new HMM(pos);
		HMM neg_clone = new HMM(neg);
		pos_clone.set_observations(wordfeats_test);
		neg_clone.set_observations(wordfeats_test);
		FKFeatures feats_test = new FKFeatures(10, pos_clone, neg_clone);
		feats_test.set_a(feats_train.get_a());
		kernel.init(feats_train, feats_test);
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
		kernel_fisher_modular x = new kernel_fisher_modular();
		run((IList)x.parameter_list[0]);
	}
}
