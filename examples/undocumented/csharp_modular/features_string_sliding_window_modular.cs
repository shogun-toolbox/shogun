using System;

public class features_string_sliding_window_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		String[] strings = new String[] {"AAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTTTTTTT"};
		StringCharFeatures f = new StringCharFeatures(strings, EAlphabet.DNA);
		f.obtain_by_sliding_window(5,1);

		DynamicIntArray positions = new DynamicIntArray();
		positions.append_element(0);
		positions.append_element(6);
		positions.append_element(16);
		positions.append_element(25);

		//f.obtain_by_position_list(8,positions);

	}
}
