using org.shogun;
using org.jblas;
public class preprocessor_classicmds_modular
{
	static preprocessor_classicmds_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Preprocessor");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double width = 1.2;

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		ClassicMDS preprocessor = new ClassicMDS();
		preprocessor.set_target_dim(1);
		preprocessor.apply_to_feature_matrix(features);

		Features.exit_shogun();
	}
}
