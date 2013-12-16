import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWDNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_hashed_wd_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int start_order = 1;
		int hash_bits = 2;

		int from_order = order;
		StringByteFeatures f = new StringByteFeatures(RAWDNA);
		HashedWDFeatures y = new HashedWDFeatures(f,start_order,order,from_order,hash_bits);

	}
}
