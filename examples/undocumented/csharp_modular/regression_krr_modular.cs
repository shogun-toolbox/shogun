using System;

public class regression_krr_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 0.8;
		double tau = 1e-6;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		GaussianKernel kernel= new GaussianKernel(feats_train, feats_train, width);

		RegressionLabels labels = new RegressionLabels(trainlab);

		KernelRidgeRegression krr = new KernelRidgeRegression(tau, kernel, labels);
		krr.train(feats_train);

		kernel.init(feats_train, feats_test);
		double[] out_labels = LabelsFactory.to_regression(krr.apply()).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

	}
}
