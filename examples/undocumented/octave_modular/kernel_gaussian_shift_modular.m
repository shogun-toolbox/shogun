modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% gaussian_shift
disp('GaussianShift')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.8;
max_shift=2;
shift_step=1;

kernel=GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
