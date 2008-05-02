rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_test=2000;
num_a=3;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=8;
order=20;
shift=10 ;
max_mismatch=0;
cache=100;
single_degree=-1;
x=shift*rand(1,len);
%x(:)=0;
shifts = sprintf( '%i ', floor(x(end:-1:1)) );

%generate some toy data
acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
aas=floor((shift+1)*rand(num_test,1));
idx=find(testlab==1);
for i=1:length(idx),
	testdat(aa+aas(i),idx(i))='A';
end

%traindat=traindat(1:5,:) ;
%testdat=testdat(1:5,:) ;
%len=5 ;
traindat(end,end)='A' ;
%traindat'
%input('key to continue')

%train svm
sg('send_command', 'use_linadd 1' );
sg('send_command', 'use_batch_computation 1');
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS2 CHAR 10 %i %i %i %s', order, max_mismatch, len, shifts ) );
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, max_mismatch, len, shifts));
%sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
%sg('set_WD_position_weights', ones(1,100)/100) ;
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
sg('send_command', 'svm_train');

%w=sg('get_subkernel_weights') ;
%w(1:3)=1 ;
%w(2:3)=0 ;
%w(3)=1 ;
%sg('set_subkernel_weights',w) ;

sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');

  sg('send_command', 'use_batch_computation 0');
  sg('send_command', 'delete_kernel_optimization');
  out1=sg('svm_classify');
  fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))

  sg('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT') ;
  sg('send_command', 'use_batch_computation 1');
  sg('send_command', 'delete_kernel_optimization');
  out2=sg('svm_classify');
  fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))

  sg('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY') ;
  sg('send_command', 'use_batch_computation 1');
  sg('send_command', 'delete_kernel_optimization');
  out3=sg('svm_classify');
  fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))

sg('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT') ;
%sg('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY') ;
sg('send_command', 'use_batch_computation 0');
tic;sg('send_command', 'init_kernel_optimization');toc;
%sg('send_command', 'delete_kernel_optimization');
tic;out4=sg('svm_classify');toc;
fprintf('accuracy: %f                                                                                         \n', mean(sign(out4)==testlab))

sg('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY') ;
sg('send_command', 'use_batch_computation 0');
tic;sg('send_command', 'init_kernel_optimization');toc;
%sg('send_command', 'delete_kernel_optimization');
tic;out5=sg('svm_classify');toc;
fprintf('accuracy: %f                                                                                         \n', mean(sign(out5)==testlab))


max(abs(out1-out2))
max(abs(out1-out3))
max(abs(out1-out4))
max(abs(out1-out5))
%max(abs(out2-out3))
%xmax(abs(out3-out4))
return

%evaluate svm on train data
sg('set_features', 'TEST', traindat,'DNA');
sg('set_labels', 'TEST', trainlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))

%evaluate svm on test data
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==testlab))
