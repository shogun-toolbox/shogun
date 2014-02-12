import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;

public class tests_check_commwordkernel_memleak_modular {
	static {
		System.loadLibrary("modshogun");
	}
	public static String repeat(String toRepeat, int num) {
		StringBuilder repeated = new StringBuilder(toRepeat.length() * num);
		for (int i = 0; i < num; i++)
			repeated.append(toRepeat);
		return repeated.toString();
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int num = 10;
		int order = 7;
		int gap = 0;
		boolean reverse = false;

		String POS[] = new String[141];
		for (int i = 0; i < 60; i++) {
			POS[i] = repeat("ACGT", 10);
		}
		for (int i = 60; i < 82; i++) {
			POS[i] = repeat("TTGT", 10);
		}
		for (int i = 82; i < 141; i++) {
			POS[i] = repeat("ACGT", 10);
		}

		String NEG[] = new String[141];
		for (int i = 0; i < 60; i++) {
			NEG[i] = repeat("ACGT", 10);
		}
		for (int i = 60; i < 82; i++) {
			NEG[i] = repeat("TTGT", 10);
		}
		for (int i = 82; i < 141; i++) {
			NEG[i] = repeat("ACGT", 10);
		}

		String POSNEG[] = new String[282];
		for (int i = 0; i < 141; i++) {
			POSNEG[i] = POS[i];
			POSNEG[i + 141] = NEG[i];
		}

		for(int i = 0; i < 10; i++) {
			Alphabet alpha= new Alphabet(DNA);
			StringCharFeatures traindat = new StringCharFeatures(alpha);
			traindat.set_features(POSNEG);
			StringWordFeatures trainudat = new StringWordFeatures(traindat.get_alphabet());
			trainudat.obtain_from_char(traindat, order-1, order, gap, reverse);
			SortWordString pre = new SortWordString();
			pre.init(trainudat);
			trainudat.add_preprocessor(pre);
			trainudat.apply_preprocessor();
			CommWordStringKernel spec = new CommWordStringKernel(10, false);
			spec.set_normalizer(new IdentityKernelNormalizer());
			spec.init(trainudat, trainudat);
			DoubleMatrix K = spec.get_kernel_matrix();
		}

	}
}
