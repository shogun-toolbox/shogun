size_cache=10;
C=0.017;
use_bias=false;
epsilon=1e-5;
width=2.1;
max_train_time=60;

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% SVMOcas
disp('SVMOcas');
sg('new_classifier', 'SVMOCAS');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');

