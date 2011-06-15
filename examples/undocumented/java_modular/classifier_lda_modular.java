import org.shogun.*;
import org.jblas.*;
public class classifier_lda_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		int gamma = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../../data/toy/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../../data/toy/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../../data/toy/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		Labels labels = new Labels(trainlab);

		LDA lda = new LDA(gamma, feats_train, labels);
		lda.train();

		lda.get_bias();
		lda.get_w();
		lda.set_features(feats_test);
		DoubleMatrix out_labels = lda.apply().get_labels();

		Features.exit_shogun();
	}
}
