import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class features_string_file_char_modular {
	static {
		System.loadLibrary("Features");
	}

	public static String fname = new String();
	public features_string_file_char_modular() {
		fname = "features_string_file_char_modular.java";
	}
	static ArrayList run(String filename) {
		Features.init_shogun_with_defaults();
		StringFileCharFeatures f = new StringFileCharFeatures(fname, RAWBYTE);

		ArrayList result = new ArrayList();
		result.add(f);

		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_file_char_modular x = new features_string_file_char_modular();
		run(x.fname);
	}
}
