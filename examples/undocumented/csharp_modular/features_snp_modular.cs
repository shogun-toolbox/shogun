//import org.shogun.*;
//import org.jblas.*;
//import static org.shogun.EAlphabet.SNP;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;

public class features_snp_modular {
    public static void Main() {

	modshogun.init_shogun_with_defaults();
	string filename = "../data/snps.dat";
	StringByteFeatures sf = new StringByteFeatures(EAlphabet.SNP);
	sf.load_ascii_file(filename, false, EAlphabet.SNP, EAlphabet.SNP);
	SNPFeatures snps = new SNPFeatures(sf);


    }
}