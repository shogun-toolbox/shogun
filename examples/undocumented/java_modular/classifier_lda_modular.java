import org.shogun.*;
import org.jblas.*;

import static org.shogun.LabelsFactory.to_binary;

public class classifier_lda_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int gamma = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LDA lda = new LDA(gamma, feats_train, labels);
		lda.train();

		System.out.println(lda.get_bias());
		System.out.println(lda.get_w().toString());
		lda.set_features(feats_test);
		DoubleMatrix out_labels = to_binary(lda.apply()).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
