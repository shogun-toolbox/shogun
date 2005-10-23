C=10;
svm_eps=1e-5;
debug=0;
ORDER=10;
MISMATCH = 0 ;

num=20000;
dims=100;
numval=10000;

rand('state',sum(100*clock));
acgt='acgt' ;
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

gf('send_command', 'loglevel ALL');

gf('send_command', 'use_mkl 0') ;
gf('send_command', 'use_linadd 1') ;
gf('send_command', 'use_precompute 0') ;
gf('send_command', 'mkl_parameters 1e-5 0') ;
gf('send_command', sprintf('svm_epsilon %f', svm_eps)) ;
gf('send_command', 'clean_features TRAIN') ;
gf('send_command', 'clean_kernels') ;

gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR 10 %i %i', ORDER, MISMATCH)) ;

gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'new_svm LIGHT');
gf('send_command', sprintf('c %f', C));
tic;gf('send_command', 'svm_train');train_time=toc

[b, alpha_tmp]=gf('get_svm');
tic;gf('send_command', 'init_kernel_optimization') ;opt_time=toc

gf('set_features', 'TEST', valdat);
gf('send_command', 'init_kernel TEST');

tic;out=gf('svm_classify');val_time=toc
valerr=mean(vallab~=sign(out));
valerr
