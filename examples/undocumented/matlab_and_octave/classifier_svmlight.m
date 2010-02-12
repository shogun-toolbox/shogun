C=1.2;
use_bias=false;
epsilon=1e-5;

addpath('tools');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% SVMLight
try
	disp('SVMLight');

	degree=20;
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
	sg('set_labels', 'TRAIN', label_train_dna);
	sg('new_classifier', 'SVMLIGHT');
	sg('svm_epsilon', epsilon);
	sg('svm_use_bias', use_bias);
	sg('c', C);

	sg('train_classifier');

	sg('set_features', 'TEST', fm_test_dna, 'DNA');
	result=sg('classify');
catch
	disp('No support for SVMLight available.')
end

