import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class distribution_histogram_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Distribution");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_histogram_modular() {

		parameter_list.add(Arrays.asList(new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(4), new Integer(0)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		Features.init_shogun_with_defaults();
		int order = ((Integer)para.get(0)).intValue();
		int gap = ((Integer)para.get(1)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		Histogram histo = new Histogram(feats);
		histo.train();

		histo.get_histogram();

		int  num_examples = feats.get_num_vectors();
		int num_param = histo.get_num_model_parameters();

		DoubleMatrix out_likelihood = histo.get_log_likelihood();
		double out_sample = histo.get_log_likelihood_sample();

		ArrayList result = new ArrayList();
		result.add(histo);
		result.add(out_sample);
		result.add(out_likelihood);
		Features.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		distribution_histogram_modular x = new distribution_histogram_modular();
		run((List)x.parameter_list.get(0));
	}
}
