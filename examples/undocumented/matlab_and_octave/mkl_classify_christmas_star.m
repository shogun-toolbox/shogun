% This script should enable you to rerun the experiment in the
% paper that we labeled with "christmas star".
%
% The task is to classify two star-shaped classes that share the
% midpoint. The difficulty of the learning problem depends on the
% distance between the classes, which is varied
%
% Our model selection leads to a choice of C = 0.5. The model
% selection is not repeated inside this script.


% Preliminary settings:

C = 0.5;         % SVM Parameter
cache_size = 50; % cache per kernel in MB
svm_eps=1e-3;	 % svm epsilon
mkl_eps=1e-3;	 % mkl epsilon

no_obs = 50;     % number of observations / data points (sum for train and test and both classes)
				 % 2000 was used in the paper
k_star = 20;     % number of "leaves" of the stars
alpha = 0.3;     % noise level of the data

radius_star(:,1) = [4.1:0.2:10]';    % increasing radius of the 1.class
radius_star(:,2) = 4*ones(length(radius_star(:,1)),1);   % fixed radius 2.class
                                     % distanz between the classes: diff(radius_star(:,1)-radius_star(:,2))
rbf_width = [0.01 0.1 1 10 100];     % different width for the five used rbf kernels


mkl_norm = 1; % >=1

ent_lambda = 0;  % 0<=lambda<=1
mkl_block_norm = 4;  % 1<=blocknorm<=inf

rand('state', 17);
randn('state', 17);
%%%%
%%%% Great loop: train MKL for every data set (the different distances between the stars)
%%%%

%sg('loglevel', 'ALL');
%sg('echo', 'ON');


for kk = 1:size(radius_star,1)

  % data generation
  fprintf('MKL for radius %+02.2f                                                      \n', radius_star(kk,1))

  dummy(1,:) = rand(1,4*no_obs);
  noise = alpha*randn(1,4*no_obs);

  dummy(2,:) = sin(k_star*pi*dummy(1,:)) + noise;         % sine
  dummy(2,1:2*no_obs) = dummy(2,1:2*no_obs)+ radius_star(kk,1);         % distanz shift: first class
  dummy(2,(2*no_obs+1):end) = dummy(2,(2*no_obs+1):end)+ radius_star(kk,2); % distanz shift: second class

  dummy(1,: ) = 2*pi*dummy(1,:);

  x(1,:) =  dummy(2,:).*sin(dummy(1,:));
  x(2,:) =  dummy(2,:).*cos(dummy(1,:));

  train_y = [-ones(1,no_obs) ones(1,no_obs)];
  test_y = [-ones(1,no_obs) ones(1,no_obs)];

  train_x = x(:,1:2:end);
  test_x  = x(:,2:2:end);

  clear dummy x;

  % train MKL

  sg('clean_kernel');
  sg('clean_features', 'TRAIN');
  sg('add_features','TRAIN', train_x);       % set a trainingset for every SVM
  sg('add_features','TRAIN', train_x);
  sg('add_features','TRAIN', train_x);
  sg('add_features','TRAIN', train_x);
  sg('add_features','TRAIN', train_x);
  sg('set_labels','TRAIN', train_y);         % set the labels
  sg('new_classifier', 'MKL_CLASSIFICATION');
  sg('mkl_use_interleaved_optimization', 1); % 0, 1

  % use standard (p-norm) MKL
  sg('set_solver', 'DIRECT'); % DIRECT, BLOCK_NORM, NEWTON, CPLEX, AUTO, GLPK, ELASTICNET
  sg('mkl_parameters', mkl_eps, 0, mkl_norm);

  % elastic net MKL
  %sg('set_solver', 'ELASTICNET');
  %sg('elasticnet_lambda',ent_lambda);

  % mixed norm MKL
  %sg('set_solver', 'BLOCK_NORM');
  %sg('mkl_block_norm', mkl_block_norm);

  %sg('set_constraint_generator', 'LIBSVM');
  sg('svm_epsilon', svm_eps);
  sg('set_kernel', 'COMBINED', 0);
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(1));
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(2));
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(3));
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(4));
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(5));
  sg('c', C);
  sg('train_classifier');
  [b,alphas]=sg('get_svm') ;
  w(kk,:) = sg('get_subkernel_weights');

  % calculate train error

  sg('clean_features', 'TEST');
  sg('add_features','TEST',train_x);
  sg('add_features','TEST',train_x);
  sg('add_features','TEST',train_x);
  sg('add_features','TEST',train_x);
  sg('add_features','TEST',train_x);
  sg('set_labels','TEST', train_y);
  sg('set_threshold', 0);
  result.trainout(kk,:)=sg('classify');
  result.trainerr(kk)  = mean(train_y~=sign(result.trainout(kk,:)),2);

  % calculate test error

  sg('clean_features', 'TEST');
  sg('add_features','TEST',test_x);
  sg('add_features','TEST',test_x);
  sg('add_features','TEST',test_x);
  sg('add_features','TEST',test_x);
  sg('add_features','TEST',test_x);
  sg('set_labels','TEST',test_y);
  sg('set_threshold', 0);
  result.testout(kk,:)=sg('classify');
  result.testerr(kk)  = mean(test_y~=sign(result.testout(kk,:)),2);

end
disp('done. now w contains the kernel weightings and result test/train outputs and errors')
