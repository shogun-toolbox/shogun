size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% Sparse Linear
disp('SparseLinear');

scale=1.3;

sg('set_kernel', 'LINEAR', 'SPARSEREAL', size_cache, scale);

sg('set_features', 'TRAIN', sparse(fm_train_real));
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', sparse(fm_test_real));
km=sg('get_kernel_matrix', 'TEST');
