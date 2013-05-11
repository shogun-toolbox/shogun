import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.shogun.EAlphabet.DNA;
import static org.shogun.LabelsFactory.to_binary;

public class classifier_svmlight_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		int degree = 20;
		modshogun.init_shogun_with_defaults();
		double C = 1.1;
		double epsilon = 1e-5;
		int num_threads = 1;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		BinaryLabels labels = new BinaryLabels(Load.load_labels("../data/label_train_dna.dat"));
		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		kernel.init(feats_train, feats_test);
		to_binary(svm.apply()).get_labels();

		modshogun.exit_shogun();
	}
}
