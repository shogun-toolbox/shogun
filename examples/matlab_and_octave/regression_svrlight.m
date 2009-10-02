size_cache=10;
width=2.1;
C=1.2;
tube_epsilon=1e-2;

addpath('tools');
label_train=load_matrix('../data/label_train_twoclass.dat');
fm_train=load_matrix('../data/fm_train_real.dat');
fm_test=load_matrix('../data/fm_test_real.dat');

% SVR Light
try
	disp('SVRLight');

	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
	sg('set_features', 'TRAIN', fm_train);
	sg('set_labels', 'TRAIN', label_train);
	sg('new_regression', 'SVRLIGHT');
	sg('svr_tube_epsilon', tube_epsilon);
	sg('c', C);

	sg('train_regression');

	sg('set_features', 'TEST', fm_test);
	result=sg('classify');
catch
	disp('No support for SVRLight available.')
end

