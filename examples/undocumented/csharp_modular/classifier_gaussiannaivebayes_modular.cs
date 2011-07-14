using System;

using org.shogun;
using org.jblas;
public class classifier_gaussiannaivebayes_modular
{
	static classifier_gaussiannaivebayes_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		Labels labels = new Labels(trainlab);

		GaussianNaiveBayes gnb = new GaussianNaiveBayes(feats_train, labels);
		gnb.train();
		DoubleMatrix out_labels = gnb.apply(feats_test).get_labels();
		Console.WriteLine(out_labels.ToString());

		Features.exit_shogun();
	}
}
