using System;

public class classifier_averaged_perceptron_modular{

	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double learn_rate = 1.0;
		int max_iter = 1000;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");
		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		BinaryLabels labels = new BinaryLabels(trainlab);
		AveragedPerceptron perceptron = new AveragedPerceptron(feats_train, labels);
		perceptron.set_learn_rate(learn_rate);
		perceptron.set_max_iter(max_iter);
		perceptron.train();

		perceptron.set_features(feats_test);
		double[] out_labels = LabelsFactory.to_binary(perceptron.apply()).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

	}
}
