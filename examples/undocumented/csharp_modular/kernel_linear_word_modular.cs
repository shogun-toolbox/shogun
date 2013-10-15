using System;

public class kernel_linear_word_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double scale = 1.2;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_word.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_word.dat");

		short[,] traindata_word = new short[traindata_real.GetLength(0), traindata_real.GetLength(1)];
		for (int i = 0; i < traindata_real.GetLength(0); i++){
			for (int j = 0; j < traindata_real.GetLength(1); j++)
				traindata_word[i, j] = (short)traindata_real[i, j];
		}

		short[,] testdata_word = new short[testdata_real.GetLength(0), testdata_real.GetLength(1)];
		for (int i = 0; i < testdata_real.GetLength(0); i++){
			for (int j = 0; j < testdata_real.GetLength(1); j++)
				testdata_word[i, j] = (short)testdata_real[i, j];
		}
		WordFeatures feats_train = new WordFeatures(traindata_word);
		WordFeatures feats_test = new WordFeatures(testdata_word);

		LinearKernel kernel = new LinearKernel(feats_train, feats_test);
		kernel.set_normalizer(new AvgDiagKernelNormalizer(scale));
		kernel.init(feats_train, feats_train);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		foreach(double item in km_train) {
			Console.Write(item);
		}

		foreach(double item in km_test) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}
