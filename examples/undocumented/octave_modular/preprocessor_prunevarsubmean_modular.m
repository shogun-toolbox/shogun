modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

%PruneVarSubMean
disp('PruneVarSubMean')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

preproc=PruneVarSubMean();
preproc.init(feats_train);
feats_train.add_preprocessor(preproc);
feats_train.apply_preprocessor();
feats_test.add_preprocessor(preproc);
feats_test.apply_preprocessor();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
