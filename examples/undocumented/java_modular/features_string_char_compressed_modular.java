import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class features_string_char_compressed_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Library");
		System.loadLibrary("Preprocessor");
	}

	public static String filename = new String();
	public features_string_char_compressed_modular() {
		filename = "features_string_char_compressed_modular.java";
	}
	static ArrayList run(String fname) {
		
		Features.init_shogun_with_defaults();

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

		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_char_compressed_modular x = new features_string_char_compressed_modular();
		run(x.filename);
	}
}
