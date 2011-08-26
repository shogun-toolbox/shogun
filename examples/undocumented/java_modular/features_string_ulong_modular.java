import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_ulong_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		boolean rev = false;
		int start = 0;
		int order = 2;
		int gap = 0;
		StringCharFeatures cf = new StringCharFeatures(new String[] { "hey", "guys", "string"}, RAWBYTE);
		StringUlongFeatures uf = new StringUlongFeatures(RAWBYTE);

		uf.obtain_from_char(cf, start,order,gap,rev);
		uf.set_feature_vector(new DoubleMatrix(new double[][] {{1,2,3,4,5}}), 0);

		modshogun.exit_shogun();
	}
}
