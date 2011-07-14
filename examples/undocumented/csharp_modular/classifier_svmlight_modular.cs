using System.Collections;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class classifier_svmlight_modular
{
	static classifier_svmlight_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public classifier_svmlight_modular()
	{

		parameter_list.Add(Arrays.asList(new double?(1.1), new double?(1e-5), new int?(1)));
		parameter_list.Add(Arrays.asList(new double?(1.2), new double?(1e-5), new int?(1)));
	}
	internal static ArrayList run(IList para)
	{
		int degree = 20;
		Features.init_shogun_with_defaults();
		double C = (double)((double?)para[0]);
		double epsilon = (double)((double?)para[1]);
		int num_threads = (int)((int?)para[2]);

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		string[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

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
		result.Add(kernel);
		Features.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		classifier_svmlight_modular x = new classifier_svmlight_modular();
		run((IList)x.parameter_list[0]);
	}
}
