import org.shogun.*;
import org.jblas.*;
public class kernel_distance_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Distance");
		System.loadLibrary("Kernel");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 1.7;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance();

		DistanceKernel kernel = new DistanceKernel(feats_train, feats_test, width, distance);

		DoubleMatrix dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		DoubleMatrix dm_test = distance.get_distance_matrix();

		System.out.println(dm_train.toString());
		System.out.println(dm_test.toString());

		Features.exit_shogun();
	}
}
