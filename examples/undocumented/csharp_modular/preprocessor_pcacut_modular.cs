using org.shogun;
using org.jblas;
public class preprocessor_pcacut_modular
{
	static preprocessor_pcacut_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("Features");
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("Kernel");
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("Preprocessor");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double width = 1.4;
		int size_cache = 10;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures features = new RealFeatures(traindata_real);

		PCACut preproc = new PCACut();
		preproc.init(features);
		preproc.apply_to_feature_matrix(features);

		Features.exit_shogun();
	}
}
