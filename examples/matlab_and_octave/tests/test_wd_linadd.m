C=10;
svm_eps=41e-4 ;
debug=0;
ORDER=20;
MISMATCH = 0 ;

num=10000;
dims=40 ;
numval=10000;

rand('state',sum(100*clock));
acgt='ACGT' ;
traindat=[ acgt(ceil(rand(dims,num)*4)) ] ;
trainlab=[ -ones(1,num/2) ones(1,num/2) ];
for i=find(trainlab==1)
  traindat(20:26,i)='AAAAAAA' ;
end ;
valdat=[ acgt(ceil(rand(dims,numval)*4)) ];
vallab=[ -ones(1,numval/2) ones(1,numval/2) ];
for i=find(vallab==1)
  valdat(20:26,i)='AAAAAAA' ;
end ;
%sg('loglevel', 'ALL');
sg('threads', 10);
sg('svm_qpsize', 150);

sg('use_mkl',  0);
sg('use_linadd', 1);
sg('use_precompute', 0);
sg('mkl_parameters', 1e-5, 0);
sg('svm_epsilon', svm_eps);
sg('clean_features', 'TRAIN');
sg('clean_kernel');

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', 10, ORDER, MISMATCH);

sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic;sg('train_classifier');train_time=toc

[b, alpha_tmp]=sg('get_svm');
tic;sg('init_kernel_optimization') ;opt_time=toc

sg('set_features', 'TEST', valdat, 'DNA');

tic;out=sg('classify');val_time=toc
valerr=mean(vallab~=sign(out));
valerr
