import org.shogun.*;
import org.jblas.*;
public class classifier_liblinear_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Library");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double C = 0.9;
		double epsilon = 1e-3;

		Math_init_random(17);
		DoubleMatrix traindata_real = Load.load_numbers("../../data/toy/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../../data/toy/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../../data/toy/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		Labels labels = new Labels(trainlab);

		LibLinear svm = new LibLinear(C, feats_train, labels);
		svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL);
		svm.set_epsilon(epsilon);
		svm.set_bias_enabled(true);
		svm.train();
		svm.set_features(feats_test);
		DoubleMatrix out_labels = svm.apply().get_labels();

		Features.exit_shogun();
	}
}
