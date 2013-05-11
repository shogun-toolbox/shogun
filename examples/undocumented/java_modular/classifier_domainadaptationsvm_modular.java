import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;

import static org.shogun.LabelsFactory.to_binary;

public class classifier_domainadaptationsvm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int degree = 3;
		int C = 1;

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
		svm.train();

		DomainAdaptationSVM dasvm = new DomainAdaptationSVM(C, kernel, labels, svm, 1.0);
		dasvm.train();

		DoubleMatrix out = to_binary(dasvm.apply(feats_test)).get_labels();
		modshogun.exit_shogun();
	}
}
