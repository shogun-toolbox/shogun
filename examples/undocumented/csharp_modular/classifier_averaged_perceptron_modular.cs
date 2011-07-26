using System.Collections;

using org.shogun;
using org.jblas;

public class classifier_averaged_perceptron_modular
{
	static classifier_averaged_perceptron_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public classifier_averaged_perceptron_modular()
	{
		parameter_list.Add(Arrays.asList(new double?(10), new int?(1000)));
		parameter_list.Add(Arrays.asList(new double?(10), new int?(10)));
	}
	public virtual Serializable run(IList para)
	{
		modshogun.init_shogun_with_defaults();
		double learn_rate = (double)((double?)para[0]);
		int max_iter = (int)((int?)para[1]);

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
		result.Add(perceptron);
		result.Add(out_labels);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		classifier_averaged_perceptron_modular x = new classifier_averaged_perceptron_modular();
		x.run((IList)x.parameter_list[0]);
	}
}
