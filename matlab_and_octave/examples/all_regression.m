% Explicit examples on how to use regressions

size_cache=10;
width=2.1;
C=0.017;
tube_epsilon=1e-2;
len=42;
num=20;
dist=2.3;

trainlab=[ones(1,num*2) -ones(1,num*2)];
testlab=[ones(1,num*2) -ones(1,num*2)];

traindata=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];


%
% svm-based
%

% SVR Light
disp('SVRLight');

sg('set_features', 'TRAIN', traindata);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab);
sg('new_regression', 'SVRLIGHT');
sg('svr_tube_epsilon', tube_epsilon);
sg('c', C);
sg('train_regression');

sg('set_features', 'TEST', testdata);
sg('set_labels', 'TEST', testlab);
sg('init_kernel', 'TEST');
result=sg('classify');


% LibSVR
disp('LibSVR');

sg('set_features', 'TRAIN', traindata);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab);
sg('new_regression', 'LIBSVR');
sg('svr_tube_epsilon', tube_epsilon);
sg('c', C);
sg('train_regression');

sg('set_features', 'TEST', testdata);
sg('set_labels', 'TEST', testlab);
sg('init_kernel', 'TEST');
result=sg('classify');


%
% misc
%

% KRR
disp('KRR');

sg('set_features', 'TRAIN', traindata);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab);
sg('new_regression', 'KRR');
tau=1.2;
sg('krr_tau', tau);
sg('c', C);
sg('train_regression');

sg('set_features', 'TEST', testdata);
sg('init_kernel', 'TEST');
result=sg('classify');

