size_cache=10;

addpath('tools');
fm_train_word=uint16(load_matrix('../data/fm_train_word.dat'));
fm_test_word=uint16(load_matrix('../data/fm_test_word.dat'));

% LinearWord
disp('LinearWord');

scale=1.4;

sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale);

sg('set_features', 'TRAIN', fm_train_word);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_word);
km=sg('get_kernel_matrix', 'TEST');
