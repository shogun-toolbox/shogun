using System;

public class distance_manhattenword_modular {
	public static void Main() {

		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 0;
		bool reverse = false;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");
		double[,] fm_test_real = Load.load_numbers("../data/fm_test_real.dat");

		StringCharFeatures charfeat = new StringCharFeatures(EAlphabet.DNA);
		charfeat.set_features(fm_train_dna);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();

		StringCharFeatures charfeat_test = new StringCharFeatures(EAlphabet.DNA);
		charfeat_test.set_features(fm_test_dna);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		ManhattanWordDistance distance = new ManhattanWordDistance(feats_train, feats_train);

		double[,] dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		double[,] dm_test = distance.get_distance_matrix();

		foreach(double item in dm_train) {
			Console.Write(item);
		}

		foreach(double item in dm_test) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}

