using System;

public class converter_classicisomap_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();

		double[,] data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);

		Isomap preprocessor = new Isomap();
		preprocessor.set_target_dim(1);
		preprocessor.apply(features);

		modshogun.exit_shogun();
	}
}
