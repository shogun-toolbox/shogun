import org.shogun.*;
import org.jblas.*;

public class kernel_distance_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.7;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclideanDistance distance = new EuclideanDistance();

		DistanceKernel kernel = new DistanceKernel(feats_train, feats_test, width, distance);

		DoubleMatrix dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		DoubleMatrix dm_test = distance.get_distance_matrix();

		System.out.println(dm_train.toString());
		System.out.println(dm_test.toString());

		modshogun.exit_shogun();
	}
}
