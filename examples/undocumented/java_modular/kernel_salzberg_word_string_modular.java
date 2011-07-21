import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_salzberg_word_string_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_salzberg_word_string_modular() {

		parameter_list.add(Arrays.asList(new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(5), new Integer(0)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();

		int order = ((Integer)para.get(0)).intValue();
		int gap = ((Integer)para.get(1)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, false);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, false);

		Labels labels = new Labels(Load.load_labels("../data/label_train_dna.dat"));

		PluginEstimate pie = new PluginEstimate();
		pie.set_labels(labels);
		pie.set_features(feats_train);
		pie.train();

		SalzbergWordStringKernel kernel = new SalzbergWordStringKernel(feats_train, feats_train, pie, labels);
		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		pie.set_features(feats_test);
		pie.apply().get_labels();
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		kernel_salzberg_word_string_modular x = new kernel_salzberg_word_string_modular();
		run((List)x.parameter_list.get(0));
	}
}
