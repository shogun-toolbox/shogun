using System;

public class preprocessor_locallylinearembedding_modular {
	public static void Main() {
		
		modshogun.init_shogun_with_defaults();

		double[,] data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		LocallyLinearEmbedding preprocessor = new LocallyLinearEmbedding();
		preprocessor.set_target_dim(1);
		preprocessor.apply_to_feature_matrix(features);

		modshogun.exit_shogun();
	}
}
