size_cache=10;
width=2.1;
C=1.2;
tube_epsilon=1e-2;

addpath('tools');
label_train=load_matrix('../data/label_train_regression.dat');
fm_train=load_matrix('../data/fm_train_real.dat');
fm_test=load_matrix('../data/fm_test_real.dat');

% KernelRidgeRegression
disp('KernelRidgeRegression');

tau=1.2;

sg('set_features', 'TRAIN', fm_train);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_labels', 'TRAIN', label_train);
sg('new_regression', 'KERNELRIDGEREGRESSION');
sg('krr_tau', tau);
sg('c', C);

sg('train_regression');

sg('set_features', 'TEST', fm_test);
result=sg('classify');
