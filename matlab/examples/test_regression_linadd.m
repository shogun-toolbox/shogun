C=1;
degree=20;
numtrain=100;
svm_eps=1e-5;
svm_tube=0.0001;

acgt='ACGT';
traindat=[acgt([1*ones(5,10) 2*ones(5,10) 3*ones(5,10) 4*ones(5,10)])];
trainlab=[-ones(1,20) ones(1,20)];

testdat=[acgt([4*ones(5,10) 3*ones(5,10) 2*ones(5,10) 1*ones(5,10)])];
testlab=[ones(1,20) -ones(1,20)];

sg('send_command', 'new_svm SVRLIGHT');

sg('send_command', 'use_mkl 0') ;
sg('send_command', 'use_linadd 1') ;
sg('send_command', 'use_batch_computation 1') ;
sg('send_command', 'use_precompute 0') ;
sg('send_command', 'mkl_parameters 1e-5 0') ;
sg('send_command', 'svm_epsilon 1e-5') ;
sg('send_command', 'clean_features TRAIN') ;
sg('send_command', 'clean_kernels') ;

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR 10 %i 0 0 1 0', degree));
sg('send_command', 'init_kernel TRAIN');
sg('send_command', sprintf('c %f',C));
sg('send_command', sprintf('svm_epsilon %f',svm_eps));
sg('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
tic; sg('send_command', 'svm_train'); toc;
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
