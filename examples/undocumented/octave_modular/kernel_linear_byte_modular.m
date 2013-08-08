modshogun

addpath('tools');
fm_train_byte=uint8(load_matrix('../data/fm_train_byte.dat'));
fm_test_byte=uint8(load_matrix('../data/fm_test_byte.dat'));

% linear byte
disp('LinearByte')

feats_train=ByteFeatures(RAWBYTE);
feats_train.set_feature_matrix(fm_train_byte);

feats_test=ByteFeatures(RAWBYTE);
feats_test.set_feature_matrix(fm_test_byte);

kernel=LinearKernel(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
