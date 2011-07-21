import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class classifier_svmlight_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public classifier_svmlight_modular() {

		parameter_list.add(Arrays.asList(new Double(1.1), new Double(1e-5), new Integer(1)));
		parameter_list.add(Arrays.asList(new Double(1.2), new Double(1e-5), new Integer(1)));
	}
	static ArrayList run(List para) {
		int degree = 20;
		modshogun.init_shogun_with_defaults();
		double C = ((Double)para.get(0)).doubleValue();
		double epsilon = ((Double)para.get(1)).doubleValue();
		int num_threads = ((Integer)para.get(2)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		Labels labels = new Labels(Load.load_labels("../data/label_train_dna.dat"));
		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		kernel.init(feats_train, feats_test);
		svm.apply().get_labels();

		ArrayList result = new ArrayList();
		result.add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		classifier_svmlight_modular x = new classifier_svmlight_modular();
		run((List)x.parameter_list.get(0));
	}
}
