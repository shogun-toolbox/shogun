using System;

public class preprocessor_kernelpca_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 2.0;
		double threshold = 0.05;

		double[,] data = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures features = new RealFeatures(data);

		GaussianKernel kernel = new GaussianKernel(features, features, width);

		KernelPCA preprocessor = new KernelPCA(kernel);
		preprocessor.init(features);
		preprocessor.apply_to_feature_matrix(features);

	}
}
