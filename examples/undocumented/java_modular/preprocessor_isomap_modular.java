import org.shogun.*;
import org.jblas.*;

public class preprocessor_isomap_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.2;

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		ClassicIsomap classic = new ClassicIsomap();
		classic.set_target_dim(1);
		classic.apply_to_feature_matrix(features);

		LandmarkIsomap landmark = new LandmarkIsomap();
		landmark.set_target_dim(1);
		landmark.apply_to_feature_matrix(features);

		modshogun.exit_shogun();
	}
}
