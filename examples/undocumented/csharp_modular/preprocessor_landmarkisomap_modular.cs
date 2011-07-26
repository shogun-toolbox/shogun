using org.shogun;
using org.jblas;

public class preprocessor_landmarkisomap_modular
{
	static preprocessor_landmarkisomap_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		LandmarkIsomap landmark = new LandmarkIsomap();
		landmark.set_target_dim(1);
		landmark.apply_to_feature_matrix(features);

		modshogun.exit_shogun();
	}
}
