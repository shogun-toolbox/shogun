rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_test=5000;
num_a=3;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=8;
order=20;
shift=15;
max_mismatch=0;
cache=10;
single_degree=-1;
x=shift*ones(1,len);
%x(:)=0;
shifts = sprintf( '%i ', x(end:-1:1) );

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

traindat'
input('key to continue')

%train svm
sg('send_command', 'use_linadd 1' );
sg('send_command', 'use_batch_computation 1');
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS2 CHAR 10 %i %i %i %s', order, max_mismatch, len, shifts ) );
%sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
sg('send_command', 'svm_train');

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
