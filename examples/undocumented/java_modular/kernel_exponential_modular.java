import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_exponential_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double tau_coef = 5.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);
		ExponentialKernel kernel= new ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10);

		kernel.init(feats_train, feats_train);
		DoubleMatrix km_train=kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
