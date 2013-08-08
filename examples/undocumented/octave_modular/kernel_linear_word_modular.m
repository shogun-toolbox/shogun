modshogun

addpath('tools');
fm_train_word=uint16(load_matrix('../data/fm_train_word.dat'));
fm_test_word=uint16(load_matrix('../data/fm_test_word.dat'));

% linear_word
disp('LinearWord')

feats_train=WordFeatures(fm_train_word);
feats_test=WordFeatures(fm_test_word);
do_rescale=true;
scale=1.4;

kernel=LinearKernel();
kernel.set_normalizer(AvgDiagKernelNormalizer(scale));
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

