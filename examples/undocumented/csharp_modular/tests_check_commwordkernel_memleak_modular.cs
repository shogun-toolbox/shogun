using System.Text;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;
public class tests_check_commwordkernel_memleak_modular
{
	static tests_check_commwordkernel_memleak_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Preprocessor");
	}
	public static string repeat(string toRepeat, int num)
	{
		StringBuilder repeated = new StringBuilder(toRepeat.Length * num);
		for (int i = 0; i < num; i++)
		{
			repeated.Append(toRepeat);
		}
		return repeated.ToString();
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();
		int num = 10;
		int order = 7;
		int gap = 0;
		bool reverse = false;

		string[] POS = new string[141];
		for (int i = 0; i < 60; i++)
		{
			POS[i] = repeat("ACGT", 10);
		}
		for (int i = 61; i < 82; i++)
		{
			POS[i] = repeat("TTGT", 10);
		}
		for (int i = 83; i < 141; i++)
		{
			POS[i] = repeat("ACGT", 10);
		}

		string[] NEG = new string[141];
		for (int i = 0; i < 60; i++)
		{
			NEG[i] = repeat("ACGT", 10);
		}
		for (int i = 61; i < 82; i++)
		{
			NEG[i] = repeat("TTGT", 10);
		}
		for (int i = 83; i < 141; i++)
		{
			NEG[i] = repeat("ACGT", 10);
		}

		string[] POSNEG = new string[282];
		for (int i = 0; i < 141; i++)
		{
			POSNEG[i] = POS[i];
			POSNEG[i + 141] = NEG[i];
		}

		for(int i = 0; i < 10; i++)
		{
			Alphabet alpha = new Alphabet(DNA);
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

		Features.exit_shogun();
	}
}
