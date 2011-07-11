import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class distribution_linearhmm_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Distribution");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_linearhmm_modular() {

		parameter_list.add(Arrays.asList(new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(4), new Integer(0)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		Features.init_shogun_with_defaults();
		int order = ((Integer)para.get(0)).intValue();
		int gap = ((Integer)para.get(1)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		LinearHMM hmm = new LinearHMM(feats);
		hmm.train();

		hmm.get_transition_probs();

		int  num_examples = feats.get_num_vectors();
		int num_param = hmm.get_num_model_parameters();
		for (int i = 0; i < num_examples; i++)
			for(int j = 0; j < num_param; j++) {
			hmm.get_log_derivative(j, i);
		}

		DoubleMatrix out_likelihood = hmm.get_log_likelihood();
		double out_sample = hmm.get_log_likelihood_sample();

		ArrayList result = new ArrayList();
		result.add(hmm);
		result.add(out_sample);
		result.add(out_likelihood);
		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		distribution_linearhmm_modular x = new distribution_linearhmm_modular();
		run((List)x.parameter_list.get(0));
	}
}
