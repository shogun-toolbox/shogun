import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.RAWBYTE;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_string_file_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		String fname = "features_string_char_modular.java";
		StringCharFeatures f = new StringCharFeatures(RAWBYTE);
		f.load_from_directory(".");

		CSVFile fil = new CSVFile(fname);
		f.load(fil);

		modshogun.exit_shogun();
	}
}
