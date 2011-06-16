import org.shogun.*;
import org.jblas.*;
public class classifier_perceptron_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double learn_rate = 1.0;
		int max_iter = 1000;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");
		
		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		
		Labels labels = new Labels(trainlab);
		
		Perceptron perceptron = new Perceptron(feats_train, labels);
		perceptron.set_learn_rate(learn_rate);
		perceptron.set_max_iter(max_iter);
		perceptron.train();

		perceptron.set_features(feats_test);
		DoubleMatrix out_labels = perceptron.apply().get_labels();
		System.out.println(out_labels.toString());

		Features.exit_shogun();
	}
}
