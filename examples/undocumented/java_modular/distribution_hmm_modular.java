import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.CUBE;
import static org.shogun.BaumWelchViterbiType.BW_NORMAL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class distribution_hmm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		int N = 1;
		int M = 512;
		double pseudo = 1e-5;
		int order = 3;
		int gap = 0;

		String[] fm_train_dna = Load.load_cubes("../data/fm_train_cube.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, CUBE);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM hmm = new HMM(feats, N, M, pseudo);
		hmm.train();
		hmm.baum_welch_viterbi_train(BW_NORMAL);

		int  num_examples = feats.get_num_vectors();
		int num_param = hmm.get_num_model_parameters();
		for (int i = 0; i < num_examples; i++)
			for(int j = 0; j < num_param; j++) {
			hmm.get_log_derivative(j, i);
		}

		int best_path = 0;
		int best_path_state = 0;
		for(int i = 0; i < num_examples; i++){
			best_path += hmm.best_path(i);
			for(int j = 0; j < N; j++)
				best_path_state += hmm.get_best_path_state(i, j);
		}

		DoubleMatrix lik_example = hmm.get_log_likelihood();
		double lik_sample = hmm.get_log_likelihood_sample();

		modshogun.exit_shogun();
	}
}
