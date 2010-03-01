C=1;
use_bias=false;
epsilon=1e-5;
max_train_time=60;

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% LibLinear
disp('LibLinear');
% type can be one of LIBLINEAR_L2R_LR, LIBLINEAR_L2R_L2LOSS_SVC_DUAL,
%            LIBLINEAR_L2R_L2LOSS_SVC, LIBLINEAR_L2R_L1LOSS_SVC_DUAL
sg('new_classifier', 'LIBLINEAR_L2R_LR');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');
