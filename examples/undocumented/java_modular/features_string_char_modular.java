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

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		String[] strings = new String[] { "hey","guys","i","am","a","string"};
		StringCharFeatures f = new StringCharFeatures(strings, RAWBYTE);
		f.set_feature_vector(new DoubleMatrix(new double[][] {{'t','e','s','t'}}), 0);

	}
}
