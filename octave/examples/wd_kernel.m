rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=100;
num_test=500;
num_a=2;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=5;
order=20
max_mismatch=0;
cache=100;
normalize=1;
mkl_stepsize=1;
block=0;
single_degree=-1;

%generate some toy data
acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
traindat(aa,trainlab==1)='A';

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
testdat(aa,testlab==1)='A';

traindat'
input('key to continue')

%train svm
sg('send_command', 'use_linadd 1' );
sg('send_command', 'use_batch_computation 0');
sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
tic;sg('send_command', 'svm_train');toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
%sg('send_command', 'init_kernel_optimization');
%sg('send_command', 'delete_kernel_optimization');

sg('send_command', 'use_batch_computation 0');
sg('send_command', 'delete_kernel_optimization');
out1=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))


sg('send_command', 'init_kernel TEST');
sg('send_command', 'use_batch_computation 1');
out2=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))


sg('send_command', 'use_batch_computation 0');
tic;sg('send_command', 'init_kernel_optimization');toc;
%sg('send_command', 'delete_kernel_optimization');

tic;out3=sg('svm_classify');toc;
fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))

max(abs(out1-out2))
max(abs(out1-out3))
