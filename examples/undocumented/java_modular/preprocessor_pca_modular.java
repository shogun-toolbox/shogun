import org.shogun.*;
import org.jblas.*;

public class preprocessor_pca_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.4;
		int size_cache = 10;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures features = new RealFeatures(traindata_real);

		PCA preproc = new PCA();
		preproc.init(features);
		preproc.apply_to_feature_matrix(features);

	}
}
