using System;

using org.shogun;
using org.jblas;
public class distance_tanimoto_modular
{
	static distance_tanimoto_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Distance");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		TanimotoDistance distance = new TanimotoDistance(feats_train, feats_train);

		DoubleMatrix dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		DoubleMatrix dm_test = distance.get_distance_matrix();

		Console.WriteLine(dm_train.ToString());
		Console.WriteLine(dm_test.ToString());

		Features.exit_shogun();
	}
}
