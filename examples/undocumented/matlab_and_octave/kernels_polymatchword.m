size_cache=10;

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true
use_sign=false;
normalization='FULL';


% Poly Match WordString
disp('PolyMatchWordString');

degree=2;
inhomogene=true;

sg('set_kernel', 'POLYMATCH', 'WORD', size_cache, degree, inhomogene);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
km=sg('get_kernel_matrix', 'TEST');
