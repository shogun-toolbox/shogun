using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.SNP;

public class features_snp_modular
{
	static features_snp_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string fname = new string();
	public features_snp_modular()
	{
		fname = "../data/snps.dat";
	}
	internal static ArrayList run(string filename)
	{
		filename = fname;
		modshogun.init_shogun_with_defaults();
		StringByteFeatures sf = new StringByteFeatures(SNP);
		sf.load_ascii_file(fname, false, SNP, SNP);
		SNPFeatures snps = new SNPFeatures(sf);

		ArrayList result = new ArrayList();
		result.Add(snps);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_snp_modular x = new features_snp_modular();
		run(x.fname);
	}
}
