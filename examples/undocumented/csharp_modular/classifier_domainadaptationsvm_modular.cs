using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class classifier_domainadaptationsvm_modular
{
	static classifier_domainadaptationsvm_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Classifier");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		int degree = 3;
		int C = 1;

		string[] fm_train_dna = {"CGCACGTACGTAGCTCGAT", "CGACGTAGTCGTAGTCGTA", "CGACGGGGGGGGGGTCGTA", "CGACCTAGTCGTAGTCGTA", "CGACCACAGTTATATAGTA", "CGACGTAGTCGTAGTCGTA", "CGACGTAGTTTTTTTCGTA", "CGACGTAGTCGTAGCCCCA", "CAAAAAAAAAAAAAAAATA", "CGACGGGGGGGGGGGCGTA"};
		string[] fm_test_dna = {"AGCACGTACGTAGCTCGAT", "AGACGTAGTCGTAGTCGTA", "CAACGGGGGGGGGGTCGTA", "CGACCTAGTCGTAGTCGTA", "CGAACACAGTTATATAGTA", "CGACCTAGTCGTAGTCGTA", "CGACGTGGGGTTTTTCGTA", "CGACGTAGTCCCAGCCCCA", "CAAAAAAAAAAAACCAATA", "CGACGGCCGGGGGGGCGTA"};

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);
		double[][] label_train_dna = { new double[] { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 } };
		Labels labels = new Labels(new DoubleMatrix(label_train_dna));

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.train();

		DomainAdaptationSVM dasvm = new DomainAdaptationSVM(C, kernel, labels, svm, 1.0);
		dasvm.train();

		DoubleMatrix @out = dasvm.apply(feats_test).get_labels();
		Features.exit_shogun();
	}
}
