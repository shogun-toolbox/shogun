using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class distribution_ppwm_modular
{
	static distribution_ppwm_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_ppwm_modular()
	{

		parameter_list.Add(Arrays.asList(new int?(3), new int?(0)));
		parameter_list.Add(Arrays.asList(new int?(4), new int?(0)));
	}
	internal static void run(IList para)
	{
		bool reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = (int)((int?)para[0]);
		int gap = (int)((int?)para[1]);

		string[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		PositionalPWM ppwm = new PositionalPWM();
		ppwm.set_sigma(5.0);
		ppwm.set_mean(10.0);
		DoubleMatrix pwm = new DoubleMatrix(new double[][] {{0.0, 0.5, 0.1, 1.0}, {0.0, 0.5, 0.5, 0.0}, {1.0, 0.0, 0.4, 0.0}, {0.0, 0.0, 0.0, 0.0}});
		//ppwm.set_pwm(DoubleMatrix.log(pwm));
		ppwm.compute_w(20);
		DoubleMatrix w = ppwm.get_w();
		modshogun.exit_shogun();
	}
	static void Main(string[] argv)
	{
		distribution_ppwm_modular x = new distribution_ppwm_modular();
		run((IList)x.parameter_list[0]);
	}
}
