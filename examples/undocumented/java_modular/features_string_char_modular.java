import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_char_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static String[] strings = null;
	public features_string_char_modular() {
		strings = new String[] { "hey","guys","i","am","a","string"};
	}
	static ArrayList run(String[] strs) {
		modshogun.init_shogun_with_defaults();
		StringCharFeatures f = new StringCharFeatures(strings, RAWBYTE);
		f.set_feature_vector(new DoubleMatrix(new double[][] {{'t','e','s','t'}}), 0);

		ArrayList result = new ArrayList();
		result.add(f.get_features());
		result.add(f);

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_char_modular x = new features_string_char_modular();
		run(x.strings);
	}
}
