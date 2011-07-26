using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class classifier_svmlight_linear_term_modular
{
	static classifier_svmlight_linear_term_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public classifier_svmlight_linear_term_modular()
	{

		parameter_list.Add(Arrays.asList(new double?(0.9), new double?(1e-3), new int?(1)));
		parameter_list.Add(Arrays.asList(new double?(2.3), new double?(1e-5), new int?(4)));
	}
	internal static DoubleMatrix run(IList para)
	{
		int degree = 20;
		modshogun.init_shogun_with_defaults();
		double C = (double)((double?)para[0]);
		double epsilon = (double)((double?)para[1]);
		int num_threads = (int)((int?)para[2]);

		string[] fm_train_dna = {"CGCACGTACGTAGCTCGAT", "CGACGTAGTCGTAGTCGTA", "CGACGGGGGGGGGGTCGTA", "CGACCTAGTCGTAGTCGTA", "CGACCACAGTTATATAGTA", "CGACGTAGTCGTAGTCGTA", "CGACGTAGTTTTTTTCGTA", "CGACGTAGTCGTAGCCCCA", "CAAAAAAAAAAAAAAAATA", "CGACGGGGGGGGGGGCGTA"};
		string[] fm_test_dna = {"AGCACGTACGTAGCTCGAT", "AGACGTAGTCGTAGTCGTA", "CAACGGGGGGGGGGTCGTA", "CGACCTAGTCGTAGTCGTA", "CGAACACAGTTATATAGTA", "CGACCTAGTCGTAGTCGTA", "CGACGTGGGGTTTTTCGTA", "CGACGTAGTCCCAGCCCCA", "CAAAAAAAAAAAACCAATA", "CGACGGCCGGGGGGGCGTA"};

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		WeightedDegreeStringKernel kernel = new WeightedDegreeStringKernel(feats_train, feats_train, degree);
		double[][] label_train_dna = { new double[] { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 } };
		Labels labels = new Labels(new DoubleMatrix(label_train_dna));

		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_qpsize(3);
		svm.set_linear_term(new DoubleMatrix(new double[][] {{-1,-2,-3,-4,-5,-6,-7,-8,-7,-6}}));
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix @out = svm.apply().get_labels();

		modshogun.exit_shogun();
		return @out;
	}
	static void Main(string[] argv)
	{
		classifier_svmlight_linear_term_modular x = new classifier_svmlight_linear_term_modular();
		run((IList)x.parameter_list[0]);
	}
}
