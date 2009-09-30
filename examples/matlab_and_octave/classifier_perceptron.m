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

% Perceptron
disp('Perceptron');

% a bit silly, but Perceptron does not converge with original data
num=length(label_train_twoclass)/2;
label_train_twoclass=[label_train_twoclass(1:num/2) label_train_twoclass(num+1:num+num/2)];
fm_train_real=[fm_train_real(1,1:num/2) fm_train_real(1,num+1:num+num/2); fm_train_real(2,1:num/2) fm_train_real(2,num+1:num+num/2)];
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('new_classifier', 'PERCEPTRON');
%sg('set_perceptron_parameters', 1.6, 5000);
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');
