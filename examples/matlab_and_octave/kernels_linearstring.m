size_cache=10;

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% Linear String
disp('LinearString');

sg('set_kernel', 'LINEAR', 'CHAR', size_cache);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');
