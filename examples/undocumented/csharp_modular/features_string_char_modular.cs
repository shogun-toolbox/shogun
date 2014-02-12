using System;

public class features_string_char_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		string [] strings = new string[6] { "hey","guys","i","am","a","string"};
		StringCharFeatures f = new StringCharFeatures(strings, EAlphabet.RAWBYTE);
		string [] r = f.get_features();
		foreach(string item in r) {
		  Console.WriteLine(item);
	  }

	}
}
