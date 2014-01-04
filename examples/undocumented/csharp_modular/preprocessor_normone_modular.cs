using System;

public class preprocessor_normone_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 1.4;
		int size_cache = 10;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		NormOne preproc = new NormOne();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();


		Chi2Kernel kernel = new Chi2Kernel(feats_train, feats_train, width, size_cache);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();

		//  Parse and Display km_train
		Console.Write("km_train:\n");
		int numRows = km_train.GetLength(0);
		int numCols = km_train.GetLength(1);

		for(int i = 0; i < numRows; i++){
			for(int j = 0; j < numCols; j++){
				Console.Write(km_train[i,j] +" ");
			}
			Console.Write("\n");
		}

		//  Parse and Display km_test
		Console.Write("\nkm_test:\n");
		numRows = km_test.GetLength(0);
		numCols = km_test.GetLength(1);

		for(int i = 0; i < numRows; i++){
			for(int j = 0; j < numCols; j++){
				Console.Write(km_test[i,j] +" ");
			}
			Console.Write("\n");
		}

	}
}
