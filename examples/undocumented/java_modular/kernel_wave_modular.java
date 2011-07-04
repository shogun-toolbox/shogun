import org.shogun.*;
import org.jblas.*;
public class kernel_wave_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Distance");
		System.loadLibrary("Kernel");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double theta = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		WaveKernel kernel = new WaveKernel(feats_train, feats_test, theta, distance);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		System.out.println(km_train.toString());
		System.out.println(km_test.toString());

		Features.exit_shogun();
	}
}
