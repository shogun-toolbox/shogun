cache_size=50;
C=0.000001;
numtrain=1000;
svm_eps=1e-4;
mkl_eps=1e-4;
W0=0.1;
W1=0.5;
W2=2;
W3=1;

rand('state',0);
traindat=rand(120,numtrain);
trainlab=sign(rand(1,numtrain)-0.5);
testdat=traindat;
testlab=trainlab;

kmtrain{1}=ones(numtrain,numtrain);
kmtrain{2}=ones(numtrain,numtrain);
kmtrain{3}=ones(numtrain,numtrain);
kmtrain{4}=ones(numtrain,numtrain);
kmtrain{5}=eye(numtrain,numtrain);

kmtest{1}=ones(numtrain,numtrain);
kmtest{2}=ones(numtrain,numtrain);
kmtest{3}=ones(numtrain,numtrain);
kmtest{4}=ones(numtrain,numtrain);
kmtest{5}=eye(numtrain,numtrain);

sg('new_classifier', 'SVMLIGHT');
sg('clean_features', 'TRAIN');
sg('clean_kernel') ;

sg('set_labels', 'TRAIN', trainlab);
sg('set_features','TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, W0);
kmcool=sg('get_kernel_matrix', 'TRAIN');

sg('clean_features', 'TRAIN');
sg('clean_kernel') ;
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'COMBINED', cache_size);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W1);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W2);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W3);
sg('add_kernel', 1, 'CUSTOM', kmtrain{1}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmtrain{2}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmtrain{3}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmcool, 'FULL');

sg('use_mkl', 1);
sg('mkl_parameters', mkl_eps, 0);
sg('c', C);
sg('svm_epsilon', svm_eps);
sg('train_classifier');
[b, alphas]=sg('get_svm');
ws=sg('get_subkernel_weights');

sg('clean_features', 'TEST');
sg('clean_kernel');
sg('set_kernel', 'COMBINED', cache_size);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W1);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W2);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, W3);
sg('add_kernel', 1, 'CUSTOM', kmtest{1}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmtest{2}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmtest{3}, 'FULL');
sg('add_kernel', 1, 'CUSTOM', kmcool, 'FULL');
sg('set_subkernel_weights',ws);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('set_labels', 'TEST', testlab);
out=sg('classify');
mean(sign(out)~=testlab)
