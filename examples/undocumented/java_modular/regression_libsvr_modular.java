import org.shogun.*;
import org.jblas.*;
public class regression_libsvr_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Regression");
		System.loadLibrary("Classifier");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 0.8;
		int C = 1;
		double epsilon = 1e-5;
		double tube_epsilon = 1e-2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		GaussianKernel kernel= new GaussianKernel(feats_train, feats_train, width);

		Labels labels = new Labels(trainlab);

		LibSVR svr = new LibSVR(C, epsilon, kernel, labels);
		svr.set_tube_epsilon(tube_epsilon);
		svr.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = svr.apply().get_labels();
		System.out.println(out_labels.toString());

		Features.exit_shogun();
	}
}
