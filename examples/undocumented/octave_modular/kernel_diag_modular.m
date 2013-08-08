modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% diag
disp('Diag')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
diag=23.;

kernel=DiagKernel(feats_train, feats_train, diag);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
