using System;

public class classifier_gaussiannaivebayes_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		MulticlassLabels labels = new MulticlassLabels(trainlab);

		GaussianNaiveBayes gnb = new GaussianNaiveBayes(feats_train, labels);
		gnb.train();
		double[] out_labels = LabelsFactory.to_multiclass(gnb.apply(feats_test)).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}
