import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.Serializable;

public class classifier_averaged_perceptron_modular{
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2); 
	public classifier_averaged_perceptron_modular() {
		parameter_list.add(Arrays.asList(new Double(10), new Integer(1000)));
		parameter_list.add(Arrays.asList(new Double(10), new Integer(10)));
	}
	public Serializable run(List para) {
		modshogun.init_shogun_with_defaults();
		double learn_rate = ((Double)para.get(0)).doubleValue();
		int max_iter = ((Integer)para.get(1)).intValue();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");
		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		Labels labels = new Labels(trainlab);
		AveragedPerceptron perceptron = new AveragedPerceptron(feats_train, labels);
		perceptron.set_learn_rate(learn_rate);
		perceptron.set_max_iter(max_iter);
		perceptron.train();

		perceptron.set_features(feats_test);
		DoubleMatrix out_labels = perceptron.apply().get_labels();
		ArrayList result = new ArrayList();
		result.add(perceptron);
		result.add(out_labels);

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		classifier_averaged_perceptron_modular x = new classifier_averaged_perceptron_modular();
		x.run((List)x.parameter_list.get(0));
	}
}
