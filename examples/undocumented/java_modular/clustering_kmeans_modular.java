import org.shogun.*;
import org.jblas.*;
import static org.shogun.Math.init_random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class clustering_kmeans_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int k = 3;
		init_random(17);

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		KMeans kmeans = new KMeans(k, distance);
		kmeans.train();

		DoubleMatrix out_centers = kmeans.get_cluster_centers();
		kmeans.get_radiuses();

		modshogun.exit_shogun();
	}
}
