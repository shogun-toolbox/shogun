addpath tools
rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=:00;
num_test=500;
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
acgt='CCGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end
aas=floor((shift+1)*rand(num_train,num_a));
idx=find(trainlab==-1);
for i=1:length(idx)/2,
	for j=1:num_a,
		traindat(aa(1)+aas(i,j),idx(i))='A';
	end
end
for i=length(idx)/2+1:length(idx),
	traindat(aa(1:2)+aas(i,1),idx(i))='AA';
	traindat(aa(1)+1+aas(i,2),idx(i))='A';
end

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
aas=floor((shift+1)*rand(num_test,1));
idx=find(testlab==1);
for i=1:length(idx),
	testdat(aa+aas(i),idx(i))='A';
end
aas=floor((shift+1)*rand(num_test,num_a));
idx=find(testlab==-1);
for i=1:length(idx)/2,
	for j=1:num_a,
		testdat(aa(1)+aas(i,j),idx(i))='A';
	end
end
for i=length(idx)/2+1:length(idx),
	testdat(aa(1:2)+aas(i,1),idx(i))='AA';
	testdat(aa(1)+1+aas(i,2),idx(i))='A';
end

%traindat'
%input('key to continue')
%keyboard;

%train svm
sg('send_command', 'loglevel INFO' );
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
%fprintf('accuracy: %f roc: %f                                                                                        \n', mean(sign(out)==trainlab), calcrocscore(out,trainlab))


%evaluate svm on test data
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
%fprintf('accuracy: %f roc: %f                                                                                        \n', mean(sign(out)==testlab), calcrocscore(out,testlab))

consensus=sg('get_WD_consensus')'
