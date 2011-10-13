size_cache=10;

addpath('tools');
fm_train_byte=uint8(load_matrix('../data/fm_train_byte.dat'));
fm_test_byte=uint8(load_matrix('../data/fm_test_byte.dat'));

% LinearByte is b0rked
disp('LinearByte');

sg('set_kernel', 'LINEAR', 'BYTE', size_cache);

sg('set_features', 'TRAIN', fm_train_byte, 'RAWBYTE');
%km=sg('get_kernel_matrix', 'TRAIN');
%
%sg('set_features', 'TEST', fm_test_byte, 'RAWBYTE');
%km=sg('get_kernel_matrix', 'TEST');
%
