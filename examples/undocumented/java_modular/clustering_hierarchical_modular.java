import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class clustering_hierarchical_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int merges = 3;

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclideanDistance distance = new EuclideanDistance(feats_train, feats_train);

		Hierarchical hierarchical = new Hierarchical(merges, distance);
		hierarchical.train();

		DoubleMatrix out_distance = hierarchical.get_merge_distances();
		DoubleMatrix out_cluster = hierarchical.get_cluster_pairs();

		modshogun.exit_shogun();
	}
}
