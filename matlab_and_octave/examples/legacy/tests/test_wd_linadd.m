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
sg('send_command', 'loglevel ALL');
sg('send_command', 'threads 10') ;
sg('send_command', 'svm_qpsize 150');

sg('send_command', 'use_mkl 0') ;
sg('send_command', 'use_linadd 1') ;
sg('send_command', 'use_precompute 0') ;
sg('send_command', 'mkl_parameters 1e-5 0') ;
sg('send_command', sprintf('svm_epsilon %f', svm_eps)) ;
sg('send_command', 'clean_features TRAIN') ;
sg('send_command', 'clean_kernel') ;

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR 10 %i %i', ORDER, MISMATCH)) ;

sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f', C));
tic;sg('send_command', 'svm_train');train_time=toc

[b, alpha_tmp]=sg('get_svm');
tic;sg('send_command', 'init_kernel_optimization') ;opt_time=toc

sg('set_features', 'TEST', valdat, 'DNA');
sg('send_command', 'init_kernel TEST');

tic;out=sg('svm_classify');val_time=toc
valerr=mean(vallab~=sign(out));
valerr
