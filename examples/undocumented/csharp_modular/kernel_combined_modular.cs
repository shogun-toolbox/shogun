//import org.shogun.*;
//import org.jblas.*;
//import static org.shogun.EAlphabet.DNA;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;

using System;

public class kernel_combined_modular {
    public static void Main() {
    modshogun.init_shogun_with_defaults();
    int cardinality = 2;
    int cache = 10;

    double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
    double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");
    String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
    String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

    RealFeatures subfeats_train = new RealFeatures(traindata_real);
    RealFeatures subfeats_test = new RealFeatures(testdata_real);

    CombinedKernel kernel= new CombinedKernel();
    CombinedFeatures feats_train = new CombinedFeatures();
    CombinedFeatures feats_test = new CombinedFeatures();

    GaussianKernel subkernel = new GaussianKernel(cache, 1.1);
    feats_train.append_feature_obj(subfeats_train);
    feats_test.append_feature_obj(subfeats_test);
    kernel.append_kernel(subkernel);

    StringCharFeatures subkfeats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
    StringCharFeatures subkfeats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);

    int degree = 3;

    FixedDegreeStringKernel subkernel2= new FixedDegreeStringKernel(10, degree);
    feats_train.append_feature_obj(subkfeats_train);
    feats_test.append_feature_obj(subkfeats_test);
    kernel.append_kernel(subkernel2);

    subkfeats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
    subkfeats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);
    LocalAlignmentStringKernel subkernel3 = new LocalAlignmentStringKernel(10);
    feats_train.append_feature_obj(subkfeats_train);
    feats_test.append_feature_obj(subkfeats_test);
    kernel.append_kernel(subkernel3);

    kernel.init(feats_train, feats_train);
    double[,] km_train=kernel.get_kernel_matrix();

    kernel.init(feats_train, feats_test);
    double[,] km_test=kernel.get_kernel_matrix();

    modshogun.exit_shogun();

    }
}