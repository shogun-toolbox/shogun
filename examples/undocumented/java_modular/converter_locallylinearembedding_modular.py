import org.shogun.*;
import org.jblas.*;

public class converter_locallylinearembedding_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures features = new RealFeatures(data);
		LocallyLinearEmbedding lle = new LocallyLinearEmbedding();
		lle.set_target_dim(1);
		RealFeatures embedding = lle.embed(features);

		modshogun.exit_shogun();
	}
}
