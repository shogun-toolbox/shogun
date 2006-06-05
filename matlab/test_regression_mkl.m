cache_size=50;
C=10;
numtrain=1000;
svm_eps=1e-2;
svm_tube=0.01;
W1=0.5;
W2=2;
W3=1;

rand('state',0);
%rand('state',sum(100*clock));
traindat=[sort(100*rand(1,numtrain))];
traindat=[traindat(2:490) traindat(512:1000)];
trainlab=[sin(2*traindat)];
testdat=[linspace(-10,traindat(1),30) traindat linspace(traindat(end),traindat(end)+traindat(1)+10,30)];
testlab=[sin(testdat)];

%sg('send_command', 'loglevel ALL');
%sg('send_command', 'new_svm LIBSVR');
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', 'set_kernel GAUSSIAN REAL 50 10');
%sg('send_command', 'init_kernel TRAIN');
%sg('send_command', sprintf('c %f',C));
%sg('send_command', sprintf('svm_epsilon %f',svm_eps));
%sg('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
%tic; sg('send_command', 'svm_train'); toc;
%[b, alphas]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%sg('send_command', 'init_kernel TEST');
%out=sg('svm_classify');

sg('send_command', 'new_svm SVRLIGHT');
sg('send_command', 'clean_features TRAIN' );
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('set_kernel COMBINED %d', cache_size));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W1));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W2));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W3));
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'use_mkl 1');
sg('send_command', 'use_precompute 0');
sg('send_command', 'mkl_parameters 1e-3 0');
sg('send_command', sprintf('c %f',C));
sg('send_command', sprintf('svm_epsilon %f',svm_eps));
sg('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
tic; sg('send_command', 'svm_train'); toc;
[b2, alphas2]=sg('get_svm');
sg( 'clean_features', 'TEST' );
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out2=sg('svm_classify');

clf
plot(traindat,trainlab,'b-')
hold on
plot(traindat,trainlab,'bx')

%plot(testdat,out,'r-')
%plot(testdat,out,'ro')

plot(testdat,out2,'g-')
plot(testdat,out2,'go')
ws=sg('get_subkernel_weights');
