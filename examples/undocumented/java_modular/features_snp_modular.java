import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.SNP;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class features_snp_modular {
	static {
		System.loadLibrary("Features");
	}

	public static String fname = new String();
	public features_snp_modular() {
		fname = "../data/snps.dat";
	}
	static ArrayList run(String filename) {
		filename = fname;
		Features.init_shogun_with_defaults();
		StringByteFeatures sf = new StringByteFeatures(SNP);
		sf.load_ascii_file(fname, false, SNP, SNP);
		SNPFeatures snps = new SNPFeatures(sf);

		ArrayList result = new ArrayList();
		result.add(snps);

		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_snp_modular x = new features_snp_modular();
		run(x.fname);
	}
}
