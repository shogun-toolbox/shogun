using System;

public class classifier_mpdsvm_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		//  already tried double[,]
		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		BinaryLabels labels = new BinaryLabels(trainlab);

		MPDSVM svm = new MPDSVM(C, kernel, labels);
		svm.set_epsilon(epsilon);
		svm.train();

		kernel.init(feats_train, feats_test);
		//  already tried double[,]
		double[] out_labels = LabelsFactory.to_binary(svm.apply()).get_labels();

		foreach (double item in out_labels)
		      Console.Write(item);

	}
}
