using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_ulong_modular
{
	static features_string_ulong_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public features_string_ulong_modular()
	{
		parameter_list.Add(Arrays.asList(new int?(0), new int?(2), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(0), new int?(3), new int?(0)));
	}
	internal static ArrayList run(IList para)
	{
		modshogun.init_shogun_with_defaults();
		bool rev = false;
		int start = (int)((int?)para[0]);
		int order = (int)((int?)para[1]);
		int gap = (int)((int?)para[2]);
		StringCharFeatures cf = new StringCharFeatures(new string[] { "hey", "guys", "string"}, RAWBYTE);
		StringUlongFeatures uf = new StringUlongFeatures(RAWBYTE);

		uf.obtain_from_char(cf, start,order,gap,rev);
		uf.set_feature_vector(new DoubleMatrix(new double[][] {{1,2,3,4,5}}), 0);

		ArrayList result = new ArrayList();
		result.Add(uf.get_features());
		result.Add(uf.get_feature_vector(2));
		result.Add(uf.get_num_vectors());

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_ulong_modular x = new features_string_ulong_modular();
		run((IList)x.parameter_list[0]);
	}
}
