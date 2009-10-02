size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% Const
disp('Const');

c=23;

sg('set_kernel', 'CONST', 'REAL', size_cache, c);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');
