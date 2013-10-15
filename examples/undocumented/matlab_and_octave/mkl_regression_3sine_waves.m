% This script should enable you to rerun the experiment in the
% paper that we labeled "mixture linear and sine ".
%
% The task is to learn a regression function where the true function
% is given by a mixture of 2 sine waves in addition to a linear trend.
% We vary the frequency of the second higher frequency sine wave.

% Setup: MKL on 10 RBF kernels of different widths on 1000 examples


% Preliminary setting

% kernel width for 10 basic SVMs
rbf_width(1) = 0.001;
rbf_width(2) = 0.005;
rbf_width(3) = 0.01;
rbf_width(4) = 0.05;
rbf_width(5) = 0.1;
rbf_width(6) = 1;
rbf_width(7) = 10;
rbf_width(8) = 50;
rbf_width(9) = 100;
rbf_width(10) = 1000;

mkl_norm = 1; % >=1

% SVM parameter
C          = 1;
cache_size = 50;
mkl_eps    = 1e-4;
svm_eps    = 1e-4;
svr_tube   = 0.01;
debug = 0;

% data
f = [0:20];  % parameter that varies the frequency of the second sine wave
no_obs = 20;    % number of observations

if debug
	sg('loglevel', 'ALL');
	sg('echo', 'ON');
else
	sg('loglevel', 'ERROR');
	sg('echo', 'OFF');
end

for kk = 1:length(f)   % Big loop
	% data generation

	train_x = [0:((4*pi)/(no_obs-1)):4*pi];
	trend = 2 * train_x* ((pi)/(max(train_x)-min(train_x)));
	wave1 = sin(train_x);
	wave2 = sin(f(kk)*train_x);
	train_y = trend + wave1 + wave2;

	% MKL learning

	kernels={};

	sg('new_classifier', 'MKL_REGRESSION');
	sg('mkl_parameters', mkl_eps, 0, mkl_norm);
	sg('mkl_use_interleaved_optimization', 1); % 0, 1
	sg('set_solver', 'DIRECT'); % DIRECT, NEWTON, CPLEX, AUTO, GLPK, ELASTICNET
	sg('c', C);
	sg('svm_epsilon',svm_eps);
	sg('svr_tube_epsilon',svr_tube);
	sg('clean_features', 'TRAIN');
	sg('clean_kernel');

	sg('set_labels', 'TRAIN', train_y);               % set labels
	sg('add_features','TRAIN', train_x);              % add features for every basic SVM
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('add_features','TRAIN', train_x);
	sg('set_kernel', 'COMBINED', 0);
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(1));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(2));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(3));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(4));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(5));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(6));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(7));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(8));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(9));
	sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(10));
	sg('train_regression');

	weights(kk,:) = sg('get_subkernel_weights') ;
	fprintf('frequency: %02.2f   rbf-kernel-weights:  %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f           \n', f(kk), weights(kk,:))
end
