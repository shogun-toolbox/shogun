modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% const
disp('Const')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
c=23.;

kernel=ConstKernel(feats_train, feats_train, c);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
