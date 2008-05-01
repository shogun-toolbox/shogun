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

% SVM parameter
C          = 1;
cache_size = 50;
mkl_eps    = 1e-4;
svm_eps    = 1e-4;
svm_tube   = 0.01;
debug = 0;

% data
f = [0:20];  % parameter that varies the frequency of the second sine wave
no_obs = 1000;    % number of observations

if debug
	sg('send_command', 'loglevel ALL');
	sg('send_command', 'echo ON');
else
	sg('send_command', 'loglevel ERROR');
	sg('send_command', 'echo OFF');
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

	sg('send_command', 'new_svm SVRLIGHT');
	sg('send_command', 'use_mkl 1');                      
	sg('send_command', 'use_precompute 0');       % precompute every SINGLE kernel!
	sg('send_command', sprintf('mkl_parameters %f 0',mkl_eps));
	sg('send_command', sprintf('c %f',C));                
	sg('send_command', sprintf('svm_epsilon %f',svm_eps));
	sg('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
	sg('send_command', 'clean_features TRAIN' );
	sg('send_command', 'clean_kernel' );

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
	sg('send_command', 'set_kernel COMBINED 0');
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(1)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(2)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(3)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(4)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(5)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(6)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(7)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(8)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(9)));
	sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, rbf_width(10)));
	sg('send_command', 'init_kernel TRAIN') ;
	sg('send_command', 'svm_train');

	weights(kk,:) = sg('get_subkernel_weights') ;
	fprintf('frequency: %02.2f   rbf-kernel-weights:  %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f %02.2f           \n', f(kk), weights(kk,:))
end
