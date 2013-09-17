%rand('state',sum(100*clock));
rand('state',123455);
%sg('loglevel', 'ALL');
num=ceil(1000*rand);
dims=ceil(10*rand);
dist=0.5;
traindat=[rand(dims,num)+dist];

testdat=[rand(dims,num)-dist rand(dims,num)+dist];
testlab=[-ones(1,num) ones(1,num)];

sg('set_features', 'TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100, 0.5);
sg('new_classifier', 'LIBSVM_ONECLASS');
sg('svm_one_class_nu', 0.1);
sg('svm_epsilon', 1e-6);
sg('train_classifier');
[b, alphas]=sg('get_svm');
o=sg('get_svm_objective');

sg('set_features', 'TEST', traindat);
train_out=sg('classify');

thresh=min(train_out)-0.01; %set wildly guessed threshold

sg('set_features', 'TEST', testdat);
out=sg('classify');
err=mean(sign(thresh-out)==testlab)

mean(out(testlab>0))
mean(train_out)

