using System;

public class distribution_linearhmm_modular {
	public static void Main() {
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 4;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
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

		double[] out_likelihood = hmm.get_log_likelihood();
		double out_sample = hmm.get_log_likelihood_sample();

		modshogun.exit_shogun();
	}
}
