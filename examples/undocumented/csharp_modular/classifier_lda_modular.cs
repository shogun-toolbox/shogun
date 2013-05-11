using System;

public class classifier_lda_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int gamma = 3;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LDA lda = new LDA(gamma, feats_train, labels);
		lda.train();

		Console.WriteLine(lda.get_bias());

		//Console.WriteLine(lda.get_w().toString());
		foreach(double item in lda.get_w()) {
			Console.Write(item);
		}


		lda.set_features(feats_test);
		double[] out_labels = LabelsFactory.to_binary(lda.apply()).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}
