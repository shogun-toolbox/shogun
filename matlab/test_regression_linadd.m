addpath ../matlab
C=1;
degree=20;
numtrain=100;
svm_eps=1e-5;
svm_tube=0.0001;

load ../matlab/dna.mat

rand('state',0);
%traindat=[sort(100*rand(1,numtrain))];
%trainlab=[sin(traindat)];
traindat=XT;
trainlab=LT;
testdat=traindat;
testlab=trainlab;

gf('send_command', 'new_svm SVRLIGHT');

gf('send_command', 'use_mkl 0') ;
gf('send_command', 'use_linadd 0') ;
gf('send_command', 'use_precompute 0') ;
gf('send_command', 'mkl_parameters 1e-5 0') ;
gf('send_command', 'svm_epsilon 1e-5') ;
gf('send_command', 'clean_features TRAIN') ;
gf('send_command', 'clean_kernels') ;

gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR 10 %i 0 0 1 0', degree));
gf('send_command', 'init_kernel TRAIN');
gf('send_command', sprintf('c %f',C));
gf('send_command', sprintf('svm_epsilon %f',svm_eps));
gf('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
tic; gf('send_command', 'svm_train'); toc;
[b, alphas]=gf('get_svm');
gf('set_features', 'TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
out=gf('svm_classify');
