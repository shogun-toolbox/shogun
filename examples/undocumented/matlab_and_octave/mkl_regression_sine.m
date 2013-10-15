% This script should enable you to rerun the experiment in the
% paper that we labeled "sine".
%
% In this regression task a sine wave is to be learned.
% We vary the frequency of the wave.

% Preliminary settings:

% Parameter for the SVMs.
C          = 10;        % obtained via model selection (not included in the script)
cache_size = 10;
mkl_eps  = 1e-4;  % threshold for precision
svm_eps    = 1e-4;
svr_tube_eps   = 1e-3;
debug = 0;

% Kernel width for the 5 "basic" SVMs
rbf_width(1) = 0.005;
rbf_width(2) = 0.05;
rbf_width(3) = 0.5;
rbf_width(4) = 1;
rbf_width(5) = 10;

mkl_norm = 1; % >=1

% data
f = [0.1:0.2:5];   % values for the different frequencies
no_obs = 100;     % number of observations

if debug
	sg('loglevel', 'ALL');
	sg('echo', 'ON');
else
	sg('loglevel', 'ERROR');
	sg('echo', 'OFF');
end

for kk = 1:length(f)    % big loop for the different learning problems

  % data generation

  train_x = [1:(((10*2*pi)-1)/(no_obs-1)):10*2*pi];
  train_y = sin(f(kk)*train_x);

  kernels={};

  % initialize MKL-SVR
  sg('new_regression', 'MKL_REGRESSION');
  sg('mkl_parameters', mkl_eps, 0, mkl_norm);
  sg('mkl_use_interleaved_optimization', 1); % 0, 1
  sg('set_solver', 'GLPK'); % DIRECT, NEWTON, CPLEX, AUTO, GLPK, ELASTICNET
  sg('c', C);
  sg('svm_epsilon', svm_eps);
  sg('svr_tube_epsilon', svr_tube_eps);
  sg('clean_features', 'TRAIN');
  sg('clean_kernel');
  sg('set_labels', 'TRAIN', train_y);               % set labels
  sg('add_features','TRAIN', train_x);              % add features for every SVR
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

  sg('train_regression');

  weights(kk,:) = sg('get_subkernel_weights') ;
  fprintf('frequency: %02.2f   rbf-kernel-weights:  %02.2f %02.2f %02.2f %02.2f %02.2f           \n', f(kk), weights(kk,:))
end
