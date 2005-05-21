C=100;
numtrain=1000;
svm_eps=1e-5;
svm_tube=0.0001;

rand('state',0);
%rand('state',sum(100*clock));
traindat=[sort(100*rand(1,numtrain))];
traindat=[traindat(2:490) traindat(512:1000)];
trainlab=[sin(traindat)];
testdat=[linspace(-10,traindat(1),30) traindat linspace(traindat(end),traindat(end)+traindat(1)+10,30)];
testlab=[sin(testdat)];

gf('send_command', 'loglevel ALL');
gf('send_command', 'new_svm LIBSVR');
gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', 'set_kernel GAUSSIAN REAL 50 10');
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

gf('send_command', 'new_svm SVRLIGHT');
gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', 'set_kernel GAUSSIAN REAL 50 10');
gf('send_command', 'init_kernel TRAIN');
gf('send_command', sprintf('c %f',C));
gf('send_command', sprintf('svm_epsilon %f',svm_eps));
gf('send_command', sprintf('svr_tube_epsilon %f',svm_tube));
tic; gf('send_command', 'svm_train'); toc;
[b2, alphas2]=gf('get_svm');
gf('set_features', 'TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
out2=gf('svm_classify');

clf
plot(traindat,trainlab,'b-')
hold on
plot(traindat,trainlab,'bx')

plot(testdat,out,'r-')
plot(testdat,out,'ro')

plot(testdat,out2,'g-')
plot(testdat,out2,'go')

