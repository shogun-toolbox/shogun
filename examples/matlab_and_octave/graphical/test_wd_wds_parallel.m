clear sg
rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1500;
num_test=1500;
num_a=3;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=8;
order=2;
shift=15;
max_mismatch=0;
normalize=1;
mkl_stepsize=1;
block=1;
cache=10;
single_degree=-1;
x=shift*ones(1,len);
x(:)=0;
shifts = int32(x(end:-1:1));

%generate some toy data
acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(40:48,idx(i))='CCCAAACCC';
end

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
aas=floor((shift+1)*rand(num_test,1));
idx=find(testlab==1);
for i=1:length(idx),
	testdat(40:48,idx(i))='CCCAAACCC';
end

%train svm
sg('threads', 4);
sg('loglevel', 'ALL');
sg('use_linadd', false);
sg('use_batch_computation', false);
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10, order, max_mismatch, len, shifts);
%sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');

%evaluate svm on test data using 4 threads vanilla kernel eval
sg('threads', 4);
sg('delete_kernel_optimization');
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out1=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))

%evaluate svm on test data using 4 threads linadd kernel eval
sg('threads', 4);
sg('init_kernel_optimization');
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out2=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))

%evaluate svm on test data using 4 threads batch kernel eval
sg('threads', 4);
sg('use_batch_computation', true);
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out3=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))

%evaluate svm on test data using 1 thread vanilla kernel eval
sg('threads', 1);
sg('delete_kernel_optimization');
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out4=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))

%evaluate svm on test data using 1 threads linadd kernel eval
sg('threads', 1);
sg('init_kernel_optimization');
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out5=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))

%evaluate svm on test data using 1 threads batch kernel eval
sg('threads',  1);
sg('use_batch_computation', true);
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out6=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))
max(abs(out1-out2))
max(abs(out1-out3))
max(abs(out1-out4))
max(abs(out1-out5))
max(abs(out1-out6))
%s=sg('get_WD_scoring',2);
%figure(2)
%clf
%imagesc(s)
%colorbar
%clear sg
