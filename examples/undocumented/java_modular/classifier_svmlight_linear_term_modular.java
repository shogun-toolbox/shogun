import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class classifier_svmlight_linear_term_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Classifier");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public classifier_svmlight_linear_term_modular() {

		parameter_list.add(Arrays.asList(new Double(0.9), new Double(1e-3), new Integer(1)));
		parameter_list.add(Arrays.asList(new Double(2.3), new Double(1e-5), new Integer(4)));
	}
	static DoubleMatrix run(List para) {
		int degree = 20;
		Features.init_shogun_with_defaults();
		double C = ((Double)para.get(0)).doubleValue();
		double epsilon = ((Double)para.get(1)).doubleValue();
		int num_threads = ((Integer)para.get(2)).intValue();

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
		Labels labels = new Labels(new DoubleMatrix(label_train_dna));

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_qpsize(3);
		svm.set_linear_term(new DoubleMatrix(new double[][] {{-1,-2,-3,-4,-5,-6,-7,-8,-7,-6}}));
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out = svm.apply().get_labels();

		Features.exit_shogun();
		return out;
	}
	public static void main(String argv[]) {
		classifier_svmlight_linear_term_modular x = new classifier_svmlight_linear_term_modular();
		run((List)x.parameter_list.get(0));
	}
}
