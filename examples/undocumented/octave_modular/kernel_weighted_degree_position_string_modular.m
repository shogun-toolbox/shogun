modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% weighted_degree_position_string
disp('WeightedDegreePositionString')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
degree=20;

kernel=WeightedDegreePositionStringKernel(feats_train, feats_train, degree);

%kernel.set_shifts(zeros(len(fm_train_dna[0]), dtype=int));

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

