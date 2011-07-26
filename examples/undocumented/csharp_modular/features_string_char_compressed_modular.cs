using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.RAWBYTE;

public class features_string_char_compressed_modular
{
	static features_string_char_compressed_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string filename = new string();
	public features_string_char_compressed_modular()
	{
		filename = "features_string_char_compressed_modular.java";
	}
	internal static ArrayList run(string fname)
	{

		modshogun.init_shogun_with_defaults();

		StringFileCharFeatures f = new StringFileCharFeatures(fname, RAWBYTE);
		//f.save_compressed("foo_uncompressed.str", UNCOMPRESSED, 1);

		StringCharFeatures f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_uncompressed.str", true);

		//f.save_compressed("foo_lzo.str", LZO, 9);
		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_lzo.str", true);

		//f.save_compressed("foo_gzip.str", GZIP, 9);
		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_gzip.str", true);

		//f.save_compressed("foo_bzip2.str", BZIP2, 9);
		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_bzip2.str", true);

		//f.save_compressed("foo_lzma.str", LZMA, 9);
		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_lzma.str", true);

		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_lzo.str", false);
		//f2.add_preprocessor(new DecompressCharString(LZO));
		f2.apply_preprocessor();

		f2 = new StringCharFeatures(RAWBYTE);
		f2.load_compressed("foo_lzo.str", false);
		//f2.add_preprocessor(new DecompressCharString(LZO));
		f2.enable_on_the_fly_preprocessing();
		ArrayList result = new ArrayList();

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_char_compressed_modular x = new features_string_char_compressed_modular();
		run(x.filename);
	}
}
