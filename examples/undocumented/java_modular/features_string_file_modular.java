import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_file_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static String fname = new String();
	public features_string_file_modular() {
		fname = "features_string_char_modular.java";
	}
	static ArrayList run(String filename) {
		modshogun.init_shogun_with_defaults();
		StringCharFeatures f = new StringCharFeatures(RAWBYTE);
		f.load_from_directory(".");

		AsciiFile fil = new AsciiFile(fname);
		f.load(fil);

		ArrayList result = new ArrayList();
		result.add(f.get_features());
		result.add(f);

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_file_modular x = new features_string_file_modular();
		run(x.fname);
	}
}
