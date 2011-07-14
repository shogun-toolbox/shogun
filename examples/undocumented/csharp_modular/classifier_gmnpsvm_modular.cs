using System;

using org.shogun;
using org.jblas;
public class classifier_gmnpsvm_modular
{
	static classifier_gmnpsvm_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		Labels labels = new Labels(trainlab);

		GMNPSVM svm = new GMNPSVM(C, kernel, labels);
		svm.set_epsilon(epsilon);
		svm.train();
		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = svm.apply(feats_test).get_labels();
		Console.WriteLine(out_labels.ToString());

		Features.exit_shogun();
	}
}
