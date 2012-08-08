import org.shogun.*;
import org.jblas.*;

public class kernel_inversemultiquadric_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double shift_coef = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		InverseMultiQuadricKernel kernel = new InverseMultiQuadricKernel(feats_train, feats_test, shift_coef, distance);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		System.out.println(km_train.toString());
		System.out.println(km_test.toString());

		modshogun.exit_shogun();
	}
}
