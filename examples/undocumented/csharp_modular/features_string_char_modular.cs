using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_char_modular
{
	static features_string_char_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string[] strings = null;
	public features_string_char_modular()
	{
		strings = new string[] { "hey","guys","i","am","a","string"};
	}
	internal static ArrayList run(string[] strs)
	{
		modshogun.init_shogun_with_defaults();
		StringCharFeatures f = new StringCharFeatures(strings, RAWBYTE);
		f.set_feature_vector(new DoubleMatrix(new double[][] {{'t','e','s','t'}}), 0);

		ArrayList result = new ArrayList();
		result.Add(f.get_features());
		result.Add(f);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_char_modular x = new features_string_char_modular();
		run(x.strings);
	}
}
