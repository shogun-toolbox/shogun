import org.shogun.*;
import org.jblas.*;

import static org.shogun.LabelsFactory.to_multiclass;

public class classifier_gaussiannaivebayes_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		MulticlassLabels labels = new MulticlassLabels(trainlab);

		GaussianNaiveBayes gnb = new GaussianNaiveBayes(feats_train, labels);
		gnb.train();
		DoubleMatrix out_labels = to_multiclass(gnb.apply(feats_test)).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
