import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_sliding_window_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static String[] strings = null;
	public features_string_sliding_window_modular() {
		strings = new String[] {"AAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTTTTTTT"};
	}
	static ArrayList run(String[] strs ) {
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
		result.add(f);

		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		features_string_sliding_window_modular x = new features_string_sliding_window_modular();
		run(x.strings);
	}
}
