size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% Poly
disp('Poly');

degree=4;
inhomogene=false;
use_normalization=true;

sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');
