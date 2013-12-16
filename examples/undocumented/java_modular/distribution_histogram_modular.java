import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class distribution_histogram_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 4;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		Histogram histo = new Histogram(feats);
		histo.train();

		DoubleMatrix histogram = histo.get_histogram();

		System.out.println(histogram);
		//int  num_examples = feats.get_num_vectors();
		//int num_param = histo.get_num_model_parameters();

		//DoubleMatrix out_likelihood = histo.get_log_likelihood();
		//double out_sample = histo.get_log_likelihood_sample();

	}
}
