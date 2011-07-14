using org.shogun;
using org.jblas;
public class preprocessor_landmarkmds_modular
{
	static preprocessor_landmarkmds_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Preprocessor");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		LandmarkMDS preprocessor = new LandmarkMDS();
		preprocessor.set_target_dim(1);
		preprocessor.apply_to_feature_matrix(features);

		Features.exit_shogun();
	}
}
