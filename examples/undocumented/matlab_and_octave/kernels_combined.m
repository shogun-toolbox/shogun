size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% Combined
disp('Combined');

sg('clean_features','TRAIN');
sg('clean_features','TEST');

sg('set_kernel', 'COMBINED', size_cache);

sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 3, false);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

km=sg('get_kernel_matrix', 'TRAIN');
km=sg('get_kernel_matrix', 'TEST');
