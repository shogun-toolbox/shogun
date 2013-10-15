% This script should enable you to rerun the experiment in the
% paper that we labeled "sine".
%
% In this regression task a sine wave is to be learned.
% We vary the frequency of the wave.

% Preliminary settings:

% Parameter for the SVMs.
C          = 1;        % obtained via model selection (not included in the script)
cache_size = 50;
mkl_eps  = 1e-6;  % threshold for precision
svm_eps    = 1e-5;
svr_tube_eps   = 1e-2;
debug = 0;

% Kernel width for the 5 "basic" SVMs
rbf_width(1) = 0.005;
rbf_width(2) = 0.05;
rbf_width(3) = 0.5;
rbf_width(4) = 1;
rbf_width(5) = 10;

% data
f = [0.1:0.2:5];   % values for the different frequencies
no_obs = 1000;     % number of observations

if debug
	sg('loglevel', 'ALL');
	sg('echo', 'ON');
else
	sg('loglevel', 'ERROR');
	sg('echo', 'OFF');
end

%for kk = 1:length(f)    % big loop for the different learning problems
for kk = 4    % big loop for the different learning problems

  % data generation

  train_x = [1:(((10*2*pi)-1)/(no_obs-1)):10*2*pi];
  train_y = sin(f(kk)*train_x);

  kernels={};

  % initialize MKL-SVR
  sg('new_regression', 'SVRLIGHT');
  sg('use_mkl', 1);
  sg('mkl_parameters', mkl_eps, 0);
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
%[b2,alphas2]=sg('get_svm');
[b,alphas]=sg('get_svm');
