import org.shogun.*;
import org.jblas.*;
public class classifier_larank_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
		System.loadLibrary("Library");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../../data/toy/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../../data/toy/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../../data/toy/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		Labels labels = new Labels(trainlab);

		LaRank svm = new LaRank(C, kernel, labels);
		svm.set_batch_mode(false);
		svm.set_epsilon(epsilon);
		svm.train();
		DoubleMatrix out_labels = svm.apply(feats_train).get_labels();

		Features.exit_shogun();
	}
}
