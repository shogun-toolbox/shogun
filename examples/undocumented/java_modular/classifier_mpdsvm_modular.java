import org.shogun.*;
import org.jblas.*;

import static org.shogun.LabelsFactory.to_binary;

public class classifier_mpdsvm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		BinaryLabels labels = new BinaryLabels(trainlab);

		MPDSVM svm = new MPDSVM(C, kernel, labels);
		svm.set_epsilon(epsilon);
		svm.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = to_binary(svm.apply()).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
