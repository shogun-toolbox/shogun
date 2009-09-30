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

% KNN
disp('KNN');

sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('new_classifier', 'KNN');

sg('train_classifier', 3);

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');

