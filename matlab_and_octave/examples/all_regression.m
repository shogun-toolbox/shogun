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
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'new_svm SVRLIGHT');
sg('send_command', sprintf('svr_tube_epsilon %f', tube_epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% LibSVR
disp('LibSVR');

sg('set_features', 'TRAIN', traindata);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'new_svm LIBSVR');
sg('send_command', sprintf('svr_tube_epsilon %f', tube_epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


%
% misc
%

% KRR
disp('KRR');

sg('set_features', 'TRAIN', traindata);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', 'new_svm KRR');
%sg('send_command', 'set_tau %f' % tau);
%sg('send_command', sprintf('c %f', C));
%sg('send_command', 'svm_train');

%sg('set_features', 'TEST', testdata);
%sg('send_command', 'init_kernel TEST');
%result=sg('svm_classify');

