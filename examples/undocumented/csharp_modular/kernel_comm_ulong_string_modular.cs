//import org.shogun.*;
//import org.jblas.*;
//import static org.shogun.EAlphabet.DNA;

using System;

public class kernel_comm_ulong_string_modular {
    public static void Main() {

	modshogun.init_shogun_with_defaults();
	int order = 3;
	int gap = 0;
	bool reverse = false;
	bool use_sign = false;

	String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
	String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

	StringCharFeatures charfeat = new StringCharFeatures(EAlphabet.DNA);
	charfeat.set_features(fm_train_dna);
	StringUlongFeatures feats_train = new StringUlongFeatures(charfeat.get_alphabet());
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
	SortUlongString preproc = new SortUlongString();
	preproc.init(feats_train);
	feats_train.add_preprocessor(preproc);
	feats_train.apply_preprocessor();

	StringCharFeatures charfeat_test = new StringCharFeatures(EAlphabet.DNA);
	charfeat_test.set_features(fm_test_dna);
	StringUlongFeatures feats_test = new StringUlongFeatures(charfeat.get_alphabet());
	feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);
	feats_test.add_preprocessor(preproc);
	feats_test.apply_preprocessor();

	CommUlongStringKernel kernel = new CommUlongStringKernel(feats_train, feats_train, use_sign);

	double[,] km_train = kernel.get_kernel_matrix();
	kernel.init(feats_train, feats_test);
	double[,] km_test = kernel.get_kernel_matrix();
	modshogun.exit_shogun();

    }
}

