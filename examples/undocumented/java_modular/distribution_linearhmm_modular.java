import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class distribution_linearhmm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 4;

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

	}
}
