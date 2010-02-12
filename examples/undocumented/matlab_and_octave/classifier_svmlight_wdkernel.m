rand('seed',17);
%sequence lengths, number of sequences
len=200;
num_train=500;
num_test=500;
num_a=2;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=5;
order=15
max_mismatch=0;
cache=100;
normalize=true;
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

%traindat'
%input('key to continue')

%train svm
sg('use_linadd', true);
sg('use_batch_computation', false);
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
%sg('set_WD_position_weights', ones(1,100)/100) ;
%sg('set_WD_position_weights', ones(1,200)/200) ;
sg('new_classifier', 'SVMLIGHT');
sg('c',C);
tic;sg('train_classifier');toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
%sg('init_kernel_optimization');
%sg('delete_kernel_optimization');

sg('use_batch_computation', true);
sg('delete_kernel_optimization');
out1=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))

sg('use_batch_computation', true);
out2=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))


sg('use_batch_computation', false);
tic;sg('init_kernel_optimization');toc;
%sg('delete_kernel_optimization');

tic;out3=sg('classify');toc;
fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))

max(abs(out1-out2))
max(abs(out1-out3))
