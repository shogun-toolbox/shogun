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

	public int[] parameter_list = new int[2];
	public clustering_kmeans_modular() {
		parameter_list[0] = 3;
		parameter_list[1] = 4;
	}
	static ArrayList run(int para) {
		modshogun.init_shogun_with_defaults();
		int k = para;
		init_random(17);

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		KMeans kmeans = new KMeans(k, distance);
		kmeans.train();

		DoubleMatrix out_centers = kmeans.get_cluster_centers();
		kmeans.get_radiuses();

		ArrayList result = new ArrayList();
		result.add(kmeans);
		result.add(out_centers);

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		clustering_kmeans_modular x = new clustering_kmeans_modular();
		run(x.parameter_list[0]);
	}
}
