% This script should enable you to rerun the experiment in the
% paper that we labeled "sine".
%
% In this regression task a sine wave is to be learned.
% We vary the frequency of the wave. 

% Preliminary settings:

% Parameter for the SVMs.
C          = 10;        % obtained via model selection (not included in the script)
cache_size = 10;
mkl_eps  = 1e-3;  % threshold for precision
svm_eps    = 1e-3;
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
      sg('send_command', 'loglevel ALL');
      sg('send_command', 'echo ON');
else
      sg('send_command', 'loglevel ERROR');
      sg('send_command', 'echo OFF');
end

for kk = 1:length(f)    % big loop for the different learning problems
  
  % data generation

  train_x = 1:(((10*2*pi)-1)/(no_obs-1)):10*2*pi;
  train_y = sin(f(kk)*train_x);
              
  kernels={};

  % initialize MKL-SVR
  sg('send_command', 'new_svm SVRLIGHT');
  sg('send_command', 'use_mkl 1');                      
  sg('send_command', 'use_precompute 3');
  sg('send_command', sprintf('mkl_parameters %f 0', mkl_eps));
  sg('send_command', sprintf('c %f',C));                
  sg('send_command', sprintf('svm_epsilon %f',svm_eps));
  sg('send_command', sprintf('svr_tube_epsilon %f',svr_tube_eps));
  sg('send_command', 'clean_features TRAIN' );
  sg('send_command', 'clean_kernels');
  sg('set_labels', 'TRAIN', train_y);               % set labels
  sg('add_features','TRAIN', train_x);              % add features for every SVR
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
  
  sg('send_command', 'init_kernel TRAIN');
  sg('send_command', 'svm_train');
       
  weights(kk,:) = sg('get_subkernel_weights') ;
  fprintf('frequency: %02.2f   rbf-kernel-weights:  %02.2f %02.2f %02.2f %02.2f %02.2f           \n', f(kk), weights(kk,:))
end

