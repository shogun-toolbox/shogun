import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;

public class kernel_oligo_string_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int size_cache = 3;
		int k = 1;
		double width = 10;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, DNA);

		OligoStringKernel kernel = new OligoStringKernel(size_cache, k, width);
		kernel.init(feats_train, feats_train);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		modshogun.exit_shogun();
	}
}
