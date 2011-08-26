import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.SNP;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_snp_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		String filename = "../data/snps.dat";
		StringByteFeatures sf = new StringByteFeatures(SNP);
		sf.load_ascii_file(filename, false, SNP, SNP);
		SNPFeatures snps = new SNPFeatures(sf);

		modshogun.exit_shogun();
	}
}
