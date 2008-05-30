% Explicit examples on how to use regressions

size_cache=10;
width=2.1;
C=0.017;
tube_epsilon=1e-2;

addpath('tools');
label_train=load_matrix('../data/label_train_oneclass.dat');
label_test=label_train;
fm_train=load_matrix('../data/fm_train_real.dat');
fm_test=load_matrix('../data/fm_test_real.dat');


%
% svm-based
%

% SVR Light
try
	disp('SVRLight');

	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
	sg('set_features', 'TRAIN', fm_train);
	sg('set_labels', 'TRAIN', label_train);
	sg('new_regression', 'SVRLIGHT');
	sg('svr_tube_epsilon', tube_epsilon);
	sg('c', C);

	sg('init_kernel', 'TRAIN');
	sg('train_regression');

	sg('set_features', 'TEST', fm_test);
	sg('set_labels', 'TEST', label_test);
	sg('init_kernel', 'TEST');
	result=sg('classify');
catch
	disp('No support for SVRLight available.')
end


% LibSVR
disp('LibSVR');

sg('set_features', 'TRAIN', fm_train);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_labels', 'TRAIN', label_train);
sg('new_regression', 'LIBSVR');
sg('svr_tube_epsilon', tube_epsilon);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_regression');

sg('set_features', 'TEST', fm_test);
sg('set_labels', 'TEST', label_test);
sg('init_kernel', 'TEST');
result=sg('classify');


%
% misc
%

% KRR
disp('KRR');

tau=1.2;

sg('set_features', 'TRAIN', fm_train);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('set_labels', 'TRAIN', label_train);
sg('new_regression', 'KRR');
sg('krr_tau', tau);
sg('c', C);

sg('init_kernel', 'TRAIN');
sg('train_regression');

sg('set_features', 'TEST', fm_test);
sg('init_kernel', 'TEST');
result=sg('classify');

