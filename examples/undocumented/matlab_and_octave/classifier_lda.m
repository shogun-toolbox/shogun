addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% LDA
disp('LDA');

sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('new_classifier', 'LDA');

sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');
