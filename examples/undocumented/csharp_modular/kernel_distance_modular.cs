using System;

public class kernel_distance_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 1.7;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclideanDistance distance = new EuclideanDistance();

		DistanceKernel kernel = new DistanceKernel(feats_train, feats_test, width, distance);

		double[,] dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		double[,] dm_test = distance.get_distance_matrix();

		//  Parse and Display km_train
		Console.Write("dm_train:\n");
		int numRows = dm_train.GetLength(0);
		int numCols = dm_train.GetLength(1);

		for(int i = 0; i < numRows; i++){
			for(int j = 0; j < numCols; j++){
				Console.Write(dm_train[i,j] +" ");
			}
			Console.Write("\n");
		}

		//  Parse and Display km_test
		Console.Write("\ndm_test:\n");
		numRows = dm_test.GetLength(0);
		numCols = dm_test.GetLength(1);

		for(int i = 0; i < numRows; i++){
			for(int j = 0; j < numCols; j++){
				Console.Write(dm_test[i,j] +" ");
			}
			Console.Write("\n");
		}


		modshogun.exit_shogun();
	}
}
