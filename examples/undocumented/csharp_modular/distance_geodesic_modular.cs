using System;

public class distance_geodesic_modular {
	public static void Main() {

		modshogun.init_shogun_with_defaults();

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		GeodesicMetric distance = new GeodesicMetric(feats_train, feats_train);

		double[,] dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		double[,] dm_test = distance.get_distance_matrix();

		foreach(double item in dm_train) {
			Console.Write(item);
		}

		foreach(double item in dm_test) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}
