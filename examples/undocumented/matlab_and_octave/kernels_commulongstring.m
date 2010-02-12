size_cache=10;
order=30;
gap=0;
reverse='n';
use_sign=0;
normalization='FULL';

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% Comm Ulong String
disp('CommUlongString');

sg('add_preproc', 'SORTULONGSTRING');
sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');
