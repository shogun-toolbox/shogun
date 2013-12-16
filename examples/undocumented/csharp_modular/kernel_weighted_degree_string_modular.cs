using System;

public class kernel_weighted_degree_string_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int degree = 3;

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		foreach(string item in fm_train_dna) {
			Console.WriteLine(item);
		}

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);

		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);
		double [] w = new double[degree];
		double sum = degree * (degree + 1)/2;
		for (int i = 0; i < degree; i++) {
			w[i] = (degree - i)/sum;
		}

		kernel.set_wd_weights(w);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		foreach(double item in km_train) {
			Console.Write(item);
		}

		foreach(double item in km_test) {
			Console.Write(item);
		}

	}
}
