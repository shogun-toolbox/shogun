using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_word_modular
{
	static features_string_word_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string[] strings = null;
	public features_string_word_modular()
	{
		strings = new string[] { "hey", "guys", "string"};
	}
	internal static ArrayList run(string[] strs)
	{
		modshogun.init_shogun_with_defaults();
		StringCharFeatures cf = new StringCharFeatures(strings, RAWBYTE);
		StringWordFeatures wf = new StringWordFeatures(RAWBYTE);
		wf.obtain_from_char(cf, 0, 2, 0, false);
		wf.set_feature_vector(new DoubleMatrix(new double[][] {{1,2,3,4,5}}), 0);

		ArrayList result = new ArrayList();
		//result.add(wf.get_features());
		result.Add(wf);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_word_modular x = new features_string_word_modular();
		run(x.strings);
	}
}
