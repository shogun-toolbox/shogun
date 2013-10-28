modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% fixed_degree_string
disp('FixedDegreeString')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
degree=3;

kernel=FixedDegreeStringKernel(feats_train, feats_train, degree);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

