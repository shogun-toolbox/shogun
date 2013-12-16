using System;

public class clustering_hierarchical_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int merges = 3;

		double[,] fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		Hierarchical hierarchical = new Hierarchical(merges, distance);
		hierarchical.train();

		double[] out_distance = hierarchical.get_merge_distances();
		int[,] out_cluster = hierarchical.get_cluster_pairs();

	}
}