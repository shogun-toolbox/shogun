using System;

public class kernel_auc_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 1.6;

		double[,] train_real = Load.load_numbers("../data/fm_train_real.dat");
		double[] trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(train_real);
		GaussianKernel subkernel = new GaussianKernel(feats_train, feats_train, width);

		BinaryLabels labels = new BinaryLabels(trainlab);

		AUCKernel kernel = new AUCKernel(0, subkernel);
		kernel.setup_auc_maximization(labels);

		double[,] km_train = kernel.get_kernel_matrix();

		int numRows = km_train.GetLength(0);
		int numCols = km_train.GetLength(1);

		Console.Write("km_train:\n");

		for(int i = 0; i < numRows; i++){
			for(int j = 0; j < numCols; j++){
				Console.Write(km_train[i,j] +" ");
			}
			Console.Write("\n");
		}

	}
}
