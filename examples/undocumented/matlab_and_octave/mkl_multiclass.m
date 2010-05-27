% Explicit examples on how to use the different classifiers

size_cache=10;
C=1.2;
use_bias=false;
epsilon=1e-5;
width=1.2;
mkl_eps=0.001;
mkl_norm=2;

max_train_time=600;

addpath('tools');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% MKL_MULTICLASS
disp('MKL_MULTICLASS');
sg('new_classifier', 'MKL_MULTICLASS');

disp('Combined');

sg('clean_kernel');
sg('clean_features','TRAIN');
sg('clean_features','TEST');

sg('set_kernel', 'COMBINED', size_cache);

sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 2);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('set_labels', 'TRAIN', label_train_multiclass);
sg('svm_epsilon', epsilon);
sg('c', C);
sg('mkl_parameters', mkl_eps, 0, mkl_norm);

sg('train_classifier');

result=sg('classify');
result
