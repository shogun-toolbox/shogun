import org.shogun.*;
import org.jblas.*;

public class preprocessor_logplusone_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.4;
		int size_cache = 10;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		LogPlusOne preproc = new LogPlusOne();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();


		Chi2Kernel kernel = new Chi2Kernel(feats_train, feats_train, width, size_cache);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		System.out.println(km_train.toString());
		System.out.println(km_test.toString());

		modshogun.exit_shogun();
	}
}
