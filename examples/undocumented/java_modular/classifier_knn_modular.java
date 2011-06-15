import org.shogun.*;
import org.jblas.*;
public class classifier_knn_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		int k = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../../data/toy/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../../data/toy/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../../data/toy/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		Labels labels = new Labels(trainlab);

		KNN knn = new KNN(k, distance, labels);
		knn.train();
		DoubleMatrix out_labels = knn.apply(feats_test).get_labels();

		Features.exit_shogun();
	}
}
