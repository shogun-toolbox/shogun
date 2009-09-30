size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');
fm_train_word=uint16(load_matrix('../data/fm_train_word.dat'));
fm_test_word=uint16(load_matrix('../data/fm_test_word.dat'));
fm_train_byte=uint8(load_matrix('../data/fm_train_byte.dat'));
fm_test_byte=uint8(load_matrix('../data/fm_test_byte.dat'));

% Weighted Comm Word String
disp('WeightedCommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');
