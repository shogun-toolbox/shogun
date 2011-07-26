using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWDNA;

public class features_string_hashed_wd_modular
{
	static features_string_hashed_wd_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList();
	public features_string_hashed_wd_modular()
	{
		DoubleMatrix A = new DoubleMatrix(new double [][] {{1,2,3},{4,0,0},{0,0,0},{0,5,0},{0,0,6},{9,9,9}});
		parameter_list.Add(Arrays.asList((A), new int?(3), new int?(1), new int?(2)));
	}
	internal static ArrayList run(IList para)
	{
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		DoubleMatrix A = (DoubleMatrix)para[0];
		int order = (int)((int?)para[1]);
		int start_order = (int)((int?)para[2]);
		int hash_bits = (int)((int?)para[3]);

		int from_order = order;
		StringByteFeatures f = new StringByteFeatures(RAWDNA);
		//f.set_features(new DoubleMatrix(new double[][]{{0,1,2,3,0,1,2,3,3,2,2,1,1}}));

		HashedWDFeatures y = new HashedWDFeatures(f,start_order,order,from_order,hash_bits);
		//DoubleMatrix fm = y.get_feature_matrix();

		ArrayList result = new ArrayList();
		result.Add(y);

		modshogun.exit_shogun();

		return result;
	}
	static void Main(string[] argv)
	{
		features_string_hashed_wd_modular x = new features_string_hashed_wd_modular();
		run((IList)x.parameter_list[0]);
	}
}
