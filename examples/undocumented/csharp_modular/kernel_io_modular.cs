using System;

public class kernel_io_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 1.2;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_test, width);
		double[,] km_train = kernel.get_kernel_matrix();
		CSVFile f=new CSVFile("gaussian_train.ascii",'w');
		kernel.save(f);

		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();
		CSVFile f_test=new CSVFile("gaussian_train.ascii",'w');
		kernel.save(f_test);

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
