import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.shogun.EAlphabet.DNA;
import static org.shogun.LabelsFactory.to_binary;

public class kernel_histogram_word_string_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		boolean reverse = false;
		int order = 3;
		int gap = 0;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, false);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat, order-1, order, gap, false);

		BinaryLabels labels = new BinaryLabels(Load.load_labels("../data/label_train_dna.dat"));

		PluginEstimate pie = new PluginEstimate();
		pie.set_labels(labels);
		pie.set_features(feats_train);
		pie.train();

		HistogramWordStringKernel kernel = new HistogramWordStringKernel(feats_train, feats_train, pie);
		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		pie.set_features(feats_test);
		to_binary(pie.apply()).get_labels();
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
