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

%sg('loglevel', 'ALL');
%sg('new_regression', 'LIBSVR');
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('set_kernel', 'GAUSSIAN', 'REAL', 50, 10);
%sg('c', C);
%sg('svm_epsilon', svm_eps);
%sg('svr_tube_epsilon', svm_tube);
%tic; sg(train_regression'); toc;
%[b, alphas]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%out=sg('classify');

sg('new_regression', 'SVRLIGHT');
sg('clean_features', 'TRAIN');
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'COMBINED', cache_size);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W1);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W2);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W3);
sg('use_mkl', true);
sg('use_precompute', 0);
sg('mkl_parameters', 1e-3, 0);
sg('c', C);
sg('svm_epsilon', svm_eps);
sg('svr_tube_epsilon', svm_tube);
tic; sg('train_regression'); toc;
[b2, alphas2]=sg('get_svm');
sg( 'clean_features', 'TEST' );
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('set_labels', 'TEST', testlab);
out2=sg('classify');

clf
plot(traindat,trainlab,'b-')
hold on
plot(traindat,trainlab,'bx')

%plot(testdat,out,'r-')
%plot(testdat,out,'ro')

plot(testdat,out2,'g-')
plot(testdat,out2,'go')
ws=sg('get_subkernel_weights');
