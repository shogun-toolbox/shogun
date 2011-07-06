import org.shogun.*;
import org.jblas.*;
public class preprocessor_kernelpcacut_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Preprocessor");
		System.loadLibrary("Kernel");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 2.0;
		double threshold = 0.05;

		DoubleMatrix data = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures features = new RealFeatures(data);

		GaussianKernel kernel = new GaussianKernel(features, features, width);
		
		KernelPCACut preprocessor = new KernelPCACut(kernel, threshold);
		preprocessor.init(features);
		preprocessor.apply_to_feature_matrix(features);

		Features.exit_shogun();
	}
}
