using org.shogun;
using org.jblas;

public class preprocessor_classicisomap_modular
{
	static preprocessor_classicisomap_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		ClassicIsomap classic = new ClassicIsomap();
		classic.set_target_dim(1);
		classic.apply_to_feature_matrix(features);

		modshogun.exit_shogun();
	}
}
