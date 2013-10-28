modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% poly
disp('Poly')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
degree=4;
inhomogene=false;
use_normalization=true;

kernel=PolyKernel(
	feats_train, feats_train, degree, inhomogene, use_normalization);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

