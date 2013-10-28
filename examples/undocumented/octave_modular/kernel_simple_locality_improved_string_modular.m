modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% simple_locality_improved_string
disp('SimpleLocalityImprovedString')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
l=5;
inner_degree=5;
outer_degree=7;

kernel=SimpleLocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

