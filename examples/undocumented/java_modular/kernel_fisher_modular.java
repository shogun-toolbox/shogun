import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import static org.shogun.BaumWelchViterbiType.BW_NORMAL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_fisher_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public kernel_fisher_modular() {

		parameter_list.add(Arrays.asList(new Integer(1), new Integer(64), new Double(1e-5), new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(1), new Integer(64), new Double(1e-1), new Integer(4), new Integer(0)));
	}
	static ArrayList run(List para) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		int N = ((Integer)para.get(0)).intValue();
		int M = ((Integer)para.get(1)).intValue();
		double pseudo = ((Double)para.get(2)).doubleValue();
		int order = ((Integer)para.get(3)).intValue();
		int gap = ((Integer)para.get(4)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");
		DoubleMatrix label_train_dna = Load.load_labels("../data/label_train_dna.dat");

		
		ArrayList fm_hmm_pos_builder = new ArrayList();
		ArrayList fm_hmm_neg_builder = new ArrayList();
		for(int i = 0; i < label_train_dna.getColumns(); i++) {
			if (label_train_dna.get(i) == 1)			
				fm_hmm_pos_builder.add(fm_train_dna[i]);
			else
				fm_hmm_neg_builder.add(fm_train_dna[i]);			
		}

		int pos_size = fm_hmm_pos_builder.size();
		int neg_size = fm_hmm_neg_builder.size();
		String[] fm_hmm_pos = new String[pos_size];
		String[] fm_hmm_neg = new String[neg_size];
		for (int i = 0; i < pos_size; i++)
			fm_hmm_pos[i] = (String)fm_hmm_pos_builder.get(i);
		for (int i = 0; i < neg_size; i++)
			fm_hmm_pos[i] = (String)fm_hmm_neg_builder.get(i);

		StringCharFeatures charfeat = new StringCharFeatures(fm_hmm_pos, DNA);
		StringWordFeatures hmm_pos_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_pos_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM pos = new HMM(hmm_pos_train, N, M, pseudo);
		pos.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_hmm_neg, DNA);
		StringWordFeatures hmm_neg_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_neg_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM neg = new HMM(hmm_neg_train, N, M, pseudo);
		neg.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures wordfeats_train = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures wordfeats_test = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		pos.set_observations(wordfeats_train);
		neg.set_observations(wordfeats_train);
		FKFeatures feats_train = new FKFeatures(10, pos, neg);
		feats_train.set_opt_a(-1);
		PolyKernel kernel = new PolyKernel(feats_train, feats_train, 1, true);
		DoubleMatrix km_train = kernel.get_kernel_matrix();

		HMM pos_clone = new HMM(pos);
		HMM neg_clone = new HMM(neg);
		pos_clone.set_observations(wordfeats_test);
		neg_clone.set_observations(wordfeats_test);
		FKFeatures feats_test = new FKFeatures(10, pos_clone, neg_clone);
		feats_test.set_a(feats_train.get_a());
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);
		modshogun.exit_shogun();
		return result;
	}
	public static void main(String argv[]) {
		kernel_fisher_modular x = new kernel_fisher_modular();
		run((List)x.parameter_list.get(0));
	}
}
