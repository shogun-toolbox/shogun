using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.Math.init_random;


public class clustering_kmeans_modular
{
	static clustering_kmeans_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public int[] parameter_list = new int[2];
	public clustering_kmeans_modular()
	{
		parameter_list[0] = 3;
		parameter_list[1] = 4;
	}
	internal static ArrayList run(int para)
	{
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
		result.Add(kmeans);
		result.Add(out_centers);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		clustering_kmeans_modular x = new clustering_kmeans_modular();
		run(x.parameter_list[0]);
	}
}
