size_cache=10;
width=1.0;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% GaussianShift
disp('GaussianShift');

max_shift=2;
shift_step=1;

sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');
