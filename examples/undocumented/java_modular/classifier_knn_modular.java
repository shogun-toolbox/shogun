import org.shogun.*;
import org.jblas.*;

import static org.shogun.LabelsFactory.to_multiclass;

public class classifier_knn_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int k = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		MulticlassLabels labels = new MulticlassLabels(trainlab);

		KNN knn = new KNN(k, distance, labels);
		knn.train();
		DoubleMatrix out_labels = to_multiclass(knn.apply(feats_test)).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
