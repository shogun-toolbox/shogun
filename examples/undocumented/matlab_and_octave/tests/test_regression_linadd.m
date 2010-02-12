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

sg('new_regression', 'SVRLIGHT');

sg('use_mkl', 0);
sg('use_linadd', 1);
sg('use_batch_computation', 1);
sg('mkl_parameters', 1e-5, 0);
sg('svm_epsilon', 1e-5);
sg('clean_features', 'TRAIN');
sg('clean_kernel');

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', 10, degree, 0, 0, 1, 0);
sg('c', C);
sg('svm_epsilon', svm_eps);
sg('svr_tube_epsilon', svm_tube);
tic; sg('train_regression'); toc;
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
out=sg('classify');
