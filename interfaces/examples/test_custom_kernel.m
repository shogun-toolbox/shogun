C=1;
sep=0.5;
numtrain=5;
numtest=7;
svm_eps=1e-5;
%for i=1:1000,
rand('state',0);
%rand('state',sum(100*clock));
traindat=[rand(2,numtrain)-sep rand(2,numtrain)];
trainlab=[-ones(1,numtrain) ones(1,numtrain)];
testdat=[rand(2,numtest)-sep rand(2,numtest)];
testlab=[-ones(1,numtest) ones(1,numtest)];
testdat(1:10)=traindat(1:10);

gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', 'set_kernel GAUSSIAN REAL 50 10');
gf('send_command', 'init_kernel TRAIN');
kt=gf('get_kernel_matrix');
gf('send_command', 'new_svm LIGHT');
gf('send_command', sprintf('c %f',C));
gf('send_command', sprintf('svm_epsilon %f',svm_eps));
tic; gf('send_command', 'svm_train'); toc;
[b, alphas]=gf('get_svm');
gf('set_features', 'TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
kte=gf('get_kernel_matrix');
out=gf('svm_classify');
valerr=mean(testlab~=sign(out));


gf('send_command', 'set_kernel CUSTOM ANY 50');
gf('set_custom_kernel',kt,'FULL2DIAG');
gf('send_command', 'init_kernel TRAIN');
kt2=gf('get_kernel_matrix');
abs(kt-kt2)<1e-6
gf('set_custom_kernel',kte,'FULL');
gf('send_command', 'init_kernel TEST');
kte2=gf('get_kernel_matrix');
abs(kte-kte2)<1e-6
