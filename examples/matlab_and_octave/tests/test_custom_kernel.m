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
sg('set_kernel', 'GAUSSIAN', 'REAL', 50, 10);
kt=sg('get_kernel_matrix', 'TRAIN');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('svm_epsilon', svm_eps);
tic; sg('train_classifier'); toc;
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
kte=sg('get_kernel_matrix', 'TEST');
out=sg('classify');
valerr=mean(testlab~=sign(out));


sg('set_kernel', 'CUSTOM', kt,'FULL2DIAG');
kt2=sg('get_kernel_matrix');
abs(kt-kt2)<1e-6
max(abs(kt(:)-kt2(:)))

sg('set_kernel', 'CUSTOM', kte,'FULL');
kte2=sg('get_kernel_matrix');
abs(kte-kte2)<1e-6
max(abs(kte(:)-kte2(:)))
