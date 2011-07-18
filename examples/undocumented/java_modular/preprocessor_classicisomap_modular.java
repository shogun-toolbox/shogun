import org.shogun.*;
import org.jblas.*;
public class preprocessor_classicisomap_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Preprocessor");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 1.2;

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);

		ClassicIsomap landmark = new ClassicIsomap();
		landmark.set_target_dim(1);
		landmark.apply_to_feature_matrix(features);

		Features.exit_shogun();
	}
}
