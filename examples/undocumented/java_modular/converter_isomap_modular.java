import org.shogun.*;
import org.jblas.*;

public class converter_isomap_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		Isomap isomap = new Isomap();
		isomap.set_target_dim(1);
		isomap.set_k(6);
		isomap.set_landmark(false);
		RealFeatures embedding = isomap.embed(features);

		modshogun.exit_shogun();
	}
}
