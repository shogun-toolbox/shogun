//import org.shogun.*;
//import org.jblas.*;
//import static org.shogun.Math.init_random;
using System;

public class clustering_kmeans_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int k = 3;
		// already tried init_random(17)
		Math.init_random(17);

		double[,] fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		KMeans kmeans = new KMeans(k, distance);
		kmeans.train();

		double[,] out_centers = kmeans.get_cluster_centers();
		kmeans.get_radiuses();

		modshogun.exit_shogun();
	}
}
