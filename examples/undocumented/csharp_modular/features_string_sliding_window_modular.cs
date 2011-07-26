using System.Collections;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.shogun.EAlphabet.DNA;

public class features_string_sliding_window_modular
{
	static features_string_sliding_window_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static string[] strings = null;
	public features_string_sliding_window_modular()
	{
		strings = new string[] {"AAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTTTTTTT"};
	}
	internal static ArrayList run(string[] strs)
	{
		modshogun.init_shogun_with_defaults();
		StringCharFeatures f = new StringCharFeatures(strs, DNA);
		f.obtain_by_sliding_window(5,1);

		f.set_features(strs);
		DynamicIntArray positions = new DynamicIntArray();
		positions.append_element(0);
		positions.append_element(6);
		positions.append_element(16);
		positions.append_element(25);

		//f.obtain_by_position_list(8,positions);

		ArrayList result = new ArrayList();
		result.Add(f);

		modshogun.exit_shogun();
		return result;
	}
	static void Main(string[] argv)
	{
		features_string_sliding_window_modular x = new features_string_sliding_window_modular();
		run(x.strings);
	}
}
