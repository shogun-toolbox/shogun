import org.shogun.*;
import org.jblas.*;

import static org.shogun.LIBLINEAR_SOLVER_TYPE.L2R_L2LOSS_SVC_DUAL;
import static org.shogun.LabelsFactory.to_binary;

public class classifier_liblinear_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double C = 0.9;
		double epsilon = 1e-3;

		org.shogun.Math.init_random(17);
		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LibLinear svm = new LibLinear(C, feats_train, labels);
		svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL);
		svm.set_epsilon(epsilon);
		svm.set_bias_enabled(true);
		svm.train();
		svm.set_features(feats_test);
		DoubleMatrix out_labels = to_binary(svm.apply()).get_labels();
		System.out.println(out_labels.toString());

	}
}
