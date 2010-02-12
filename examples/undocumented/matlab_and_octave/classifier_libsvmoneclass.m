% Explicit examples on how to use the different classifiers

size_cache=10;
C=1.2;
use_bias=false;
epsilon=1e-5;
width=2.1;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% LibSVM OneClass
disp('LibSVMOneClass');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('new_classifier', 'LIBSVM_ONECLASS');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');

