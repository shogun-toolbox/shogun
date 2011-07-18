import org.shogun.*;
import org.jblas.*;
public class preprocessor_pcacut_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Preprocessor");
	}

	public static void main(String argv[]) {
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
