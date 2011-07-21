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

	public int[] parameter_list = new int[2];
	public clustering_hierarchical_modular() {
		parameter_list[0] = 3;
		parameter_list[1] = 4;
	}
	static ArrayList run(int para) {
		modshogun.init_shogun_with_defaults();
		int merges = para;

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		Hierarchical hierarchical = new Hierarchical(merges, distance);
		hierarchical.train();

		DoubleMatrix out_distance = hierarchical.get_merge_distances();
		DoubleMatrix out_cluster = hierarchical.get_cluster_pairs();

		ArrayList result = new ArrayList();
		result.add(hierarchical);
		result.add(out_distance);
		result.add(out_cluster);
		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		clustering_hierarchical_modular x = new clustering_hierarchical_modular();
		run(x.parameter_list[0]);
	}
}
