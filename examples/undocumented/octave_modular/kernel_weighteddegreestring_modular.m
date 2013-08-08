modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% weighted_degree_string
disp('WeightedDegreeString')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
degree=20;

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

%weights=arange(1,degree+1,dtype=double)[::-1]/ \
%	sum(arange(1,degree+1,dtype=double));
%kernel.set_wd_weights(weights);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

