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

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel GAUSSIAN REAL 50 10');
sg('send_command', 'init_kernel TRAIN');
kt=sg('get_kernel_matrix');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
sg('send_command', sprintf('svm_epsilon %f',svm_eps));
tic; sg('send_command', 'svm_train'); toc;
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
kte=sg('get_kernel_matrix');
out=sg('svm_classify');
valerr=mean(testlab~=sign(out));


sg('send_command', 'set_kernel CUSTOM ANY 50');
sg('set_custom_kernel',kt,'FULL2DIAG');
sg('send_command', 'init_kernel TRAIN');
kt2=sg('get_kernel_matrix');
abs(kt-kt2)<1e-6
sg('set_custom_kernel',kte,'FULL');
sg('send_command', 'init_kernel TEST');
kte2=sg('get_kernel_matrix');
abs(kte-kte2)<1e-6
