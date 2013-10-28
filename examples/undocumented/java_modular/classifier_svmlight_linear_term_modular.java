import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.shogun.EAlphabet.DNA;
import static org.shogun.LabelsFactory.to_binary;

public class classifier_svmlight_linear_term_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		int degree = 20;
		modshogun.init_shogun_with_defaults();
		double C = 0.9;
		double epsilon = 1e-3;
		int num_threads = 1;

		String[] fm_train_dna = {"CGCACGTACGTAGCTCGAT",
		      "CGACGTAGTCGTAGTCGTA",
		      "CGACGGGGGGGGGGTCGTA",
		      "CGACCTAGTCGTAGTCGTA",
		      "CGACCACAGTTATATAGTA",
		      "CGACGTAGTCGTAGTCGTA",
		      "CGACGTAGTTTTTTTCGTA",
		      "CGACGTAGTCGTAGCCCCA",
		      "CAAAAAAAAAAAAAAAATA",
		      "CGACGGGGGGGGGGGCGTA"};
		String[] fm_test_dna = {"AGCACGTACGTAGCTCGAT",
		      "AGACGTAGTCGTAGTCGTA",
		      "CAACGGGGGGGGGGTCGTA",
		      "CGACCTAGTCGTAGTCGTA",
		      "CGAACACAGTTATATAGTA",
		      "CGACCTAGTCGTAGTCGTA",
		      "CGACGTGGGGTTTTTCGTA",
		      "CGACGTAGTCCCAGCCCCA",
		      "CAAAAAAAAAAAACCAATA",
		      "CGACGGCCGGGGGGGCGTA"};

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);
		double label_train_dna[][] = {{-1,-1,-1,-1,-1,1,1,1,1,1}};
		BinaryLabels labels = new BinaryLabels(new DoubleMatrix(label_train_dna));

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_qpsize(3);
		svm.set_linear_term(new DoubleMatrix(new double[][] {{-1,-2,-3,-4,-5,-6,-7,-8,-7,-6}}));
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		kernel.init(feats_train, feats_test);
		to_binary(svm.apply()).get_labels();

		modshogun.exit_shogun();
	}
}
