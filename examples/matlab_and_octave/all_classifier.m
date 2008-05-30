% Explicit examples on how to use the different classifiers

size_cache=10;
C=0.017;
use_bias=0;
epsilon=1e-5;
width=2.1;
max_train_time=60;

addpath('tools');
label_train_oneclass=load_matrix('../data/label_train_oneclass.dat');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%
% kernel-based SVMs
%

% SVMLight
try
	disp('SVMLight');

	degree=20;
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
	sg('set_labels', 'TRAIN', label_train_dna);
	sg('new_svm', 'LIGHT');
	sg('svm_epsilon', epsilon);
	sg('svm_use_bias', use_bias);
	sg('c', C);

	sg('init_kernel', 'TRAIN');
	sg('train_classifier');

	sg('set_features', 'TEST', fm_test_dna, 'DNA');
	sg('init_kernel', 'TEST');
	result=sg('classify');
catch
	disp('No support for SVMLight available.')
end


% LibSVM
disp('LibSVM');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'LIBSVM');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% GPBTSVM
disp('GPBTSVM');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'GPBTSVM');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');

% MPDSVM
disp('MPDSVM');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'MPDSVM');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% LibSVM MultiClass
disp('LibSVMMultiClass');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_multiclass);
sg('new_svm', 'LIBSVM_MULTICLASS');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% LibSVM OneClass
disp('LibSVMOneClass');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('new_svm', 'LIBSVM_ONECLASS');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% GMNPSVM
disp('GMNPSVM');
sg('new_svm', 'GMNPSVM');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% run with batch or linadd on LibSVM;
disp('LibSVM batch');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'LIBSVM');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
sg('init_kernel', 'TEST');
result=sg('classify');

objective=sg('get_svm_objective');
sg('use_batch_computation', 1);
sg('use_linadd', 1);
result=sg('classify');


%
% SparseLinear classifier
%

% SubgradientSVM - often does not converge
disp('SubGradientSVM');

C=0.42;
sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'SUBGRADIENTSVM');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);
% sometimes does not terminate
%sg('train_classifier');

%sg('set_features', 'TEST', sparse(fm_test_real));
%result=sg('classify');

% SVMOcas
disp('SVMOcas');
sg('new_svm', 'SVMOCAS');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');

% SVMSGD
disp('SVMSGD');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'SVMSGD');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');

% LibLinear
disp('LibLinear');
sg('new_svm', 'LIBLINEAR_LR');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');

% SVMLin
disp('SVMLin');

sg('set_features', 'TRAIN', sparse(fm_train_real));
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_svm', 'SVMLIN');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('c', C);

sg('train_classifier');

sg('set_features', 'TEST', sparse(fm_test_real));
result=sg('classify');


%
% misc classifiers
%

% KNN
disp('KNN');

sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_classifier', 'KNN');

sg('init_distance', 'TRAIN');
sg('train_classifier', 3);

sg('set_features', 'TEST', fm_test_real);
sg('init_distance', 'TEST');
result=sg('classify');


% LDA
disp('LDA');

sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_classifier', 'LDA');

sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');


% Perceptron
disp('Perceptron');

% a bit silly, but Perceptron does not converge with original data
num=length(label_train_oneclass)/2;
label_train_oneclass=[label_train_oneclass(1:num/2) label_train_oneclass(num+1:num+num/2)];
fm_train_real=[fm_train_real(1,1:num/2) fm_train_real(1,num+1:num+num/2); fm_train_real(2,1:num/2) fm_train_real(2,num+1:num+num/2)];
sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_oneclass);
sg('new_classifier', 'PERCEPTRON');
%sg('set_perceptron_parameters', 1.6, 5000);
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');


