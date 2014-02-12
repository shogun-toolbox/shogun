using System;

public class distribution_histogram_modular {
	public static void Main() {
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 4;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		Histogram histo = new Histogram(feats);
		histo.train();

		double[] histogram = histo.get_histogram();

		foreach(double item in histogram) {
			Console.Write(item);
		}
		//int  num_examples = feats.get_num_vectors();
		//int num_param = histo.get_num_model_parameters();

		//double[,] out_likelihood = histo.get_log_likelihood();
		//double out_sample = histo.get_log_likelihood_sample();

	}
}