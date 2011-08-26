import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_anova_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int cardinality = 2;
		int size_cache = 5;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		ANOVAKernel kernel = new ANOVAKernel(feats_train, feats_train, cardinality, size_cache);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
