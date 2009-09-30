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
