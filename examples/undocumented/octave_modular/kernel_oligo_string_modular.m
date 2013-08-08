modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% oligo_string
disp('OligoString')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
k=3;
width=1.2;
size_cache=10;

kernel=OligoStringKernel(size_cache, k, width);
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

