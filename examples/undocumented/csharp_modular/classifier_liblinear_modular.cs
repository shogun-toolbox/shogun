using System;

public class classifier_liblinear_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double C = 0.9;
		double epsilon = 1e-3;

		Math.init_random(17);
		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LibLinear svm = new LibLinear(C, feats_train, labels);
		svm.set_liblinear_solver_type(LIBLINEAR_SOLVER_TYPE.L2R_L2LOSS_SVC_DUAL);
		svm.set_epsilon(epsilon);
		svm.set_bias_enabled(true);
		svm.train();
		svm.set_features(feats_test);
		double[] out_labels = LabelsFactory.to_binary(svm.apply()).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

	}
}
