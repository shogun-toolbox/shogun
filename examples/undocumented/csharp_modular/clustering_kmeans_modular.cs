using System.Collections;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class clustering_kmeans_modular
{
	static clustering_kmeans_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Clustering");
		System.loadLibrary("Distance");
		System.loadLibrary("Library");
	}

	public int[] parameter_list = new int[2];
	public clustering_kmeans_modular()
	{
		parameter_list[0] = 3;
		parameter_list[1] = 4;
	}
	internal static ArrayList run(int para)
	{
		Features.init_shogun_with_defaults();
		int k = para;
		Math_init_random(17);

		DoubleMatrix fm_train = Load.load_numbers("../data/fm_train_real.dat");

		RealFeatures feats_train = new RealFeatures(fm_train);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		KMeans kmeans = new KMeans(k, distance);
		kmeans.train();

		DoubleMatrix out_centers = kmeans.get_cluster_centers();
		kmeans.get_radiuses();

		ArrayList result = new ArrayList();
		result.Add(kmeans);
		result.Add(out_centers);

		Features.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		clustering_kmeans_modular x = new clustering_kmeans_modular();
		run(x.parameter_list[0]);
	}
}
