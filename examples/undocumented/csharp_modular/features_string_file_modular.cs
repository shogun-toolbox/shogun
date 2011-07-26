using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_file_modular
{
	static features_string_file_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string fname = new string();
	public features_string_file_modular()
	{
		fname = "features_string_char_modular.java";
	}
	internal static ArrayList run(string filename)
	{
		modshogun.init_shogun_with_defaults();
		StringCharFeatures f = new StringCharFeatures(RAWBYTE);
		f.load_from_directory(".");

		AsciiFile fil = new AsciiFile(fname);
		f.load(fil);

		ArrayList result = new ArrayList();
		result.Add(f.get_features());
		result.Add(f);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_file_modular x = new features_string_file_modular();
		run(x.fname);
	}
}
