using System.Collections;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class clustering_hierarchical_modular
{
	static clustering_hierarchical_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Clustering");
		System.loadLibrary("Distance");
	}

	public int[] parameter_list = new int[2];
	public clustering_hierarchical_modular()
	{
		parameter_list[0] = 3;
		parameter_list[1] = 4;
	}
	internal static ArrayList run(int para)
	{
		Features.init_shogun_with_defaults();
		int merges = para;

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		Hierarchical hierarchical = new Hierarchical(merges, distance);
		hierarchical.train();

		DoubleMatrix out_distance = hierarchical.get_merge_distances();
		DoubleMatrix out_cluster = hierarchical.get_cluster_pairs();

		ArrayList result = new ArrayList();
		result.Add(hierarchical);
		result.Add(out_distance);
		result.Add(out_cluster);
		Features.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		clustering_hierarchical_modular x = new clustering_hierarchical_modular();
		run(x.parameter_list[0]);
	}
}
