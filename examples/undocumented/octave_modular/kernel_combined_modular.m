modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% combined
disp('Combined')

kernel=CombinedKernel();
feats_train=CombinedFeatures();
feats_test=CombinedFeatures();

subkfeats_train=RealFeatures(fm_train_real);
subkfeats_test=RealFeatures(fm_test_real);
subkernel=GaussianKernel(10, 1.2);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

subkfeats_train=StringCharFeatures(DNA);
subkfeats_train.set_features(fm_train_dna);
subkfeats_test=StringCharFeatures(DNA);
subkfeats_test.set_features(fm_test_dna);
degree=3;
subkernel=FixedDegreeStringKernel(10, degree);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

subkfeats_train=StringCharFeatures(DNA);
subkfeats_train.set_features(fm_train_dna);
subkfeats_test=StringCharFeatures(DNA);
subkfeats_test.set_features(fm_test_dna);
subkernel=LocalAlignmentStringKernel(10);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

kernel.init(feats_train, feats_train);
km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

