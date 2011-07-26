using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_file_char_modular
{
	static features_string_file_char_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string fname = new string();
	public features_string_file_char_modular()
	{
		fname = "features_string_file_char_modular.java";
	}
	internal static ArrayList run(string filename)
	{
		modshogun.init_shogun_with_defaults();
		StringFileCharFeatures f = new StringFileCharFeatures(fname, RAWBYTE);

		ArrayList result = new ArrayList();
		result.Add(f);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_file_char_modular x = new features_string_file_char_modular();
		run(x.fname);
	}
}
