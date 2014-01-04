using System;

public class converter_multidimensionalscaling_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();

		double[,] data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		MultidimensionalScaling mds = new MultidimensionalScaling();
		mds.set_target_dim(1);
		mds.set_landmark(false);

		mds.apply(features);

	}
}

