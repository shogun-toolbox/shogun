import org.shogun.*;
import org.jblas.*;

public class preprocessor_landmarkisomap_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		LandmarkIsomap landmark = new LandmarkIsomap();
		landmark.set_target_dim(1);
		landmark.apply_to_feature_matrix(features);

		System.out.println(features.get_feature_matrix());
	
		modshogun.exit_shogun();
	}
}
