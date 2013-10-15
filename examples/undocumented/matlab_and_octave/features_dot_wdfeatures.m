C=1;
order=6;
degree=order;
from_order=6;
max_mismatch=0;
cache=100;
normalize=1;
mkl_stepsize=1;
block=1;
single_degree=-1;
epsilon=1e-5;


rand('seed',17);
%sequence lengths, number of sequences
len=20;
num_train=10;
num_a=5;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));
epsilon=1e-6;

%generate some toy data
acgt='ACGT';
shift=1;
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end

testdat=traindat;
testlab=trainlab;

%train svm
sg('threads',1);
sg('use_linadd', 1);
sg('use_batch_computation', 1);
sg('progress', 'ON');
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('svm_use_bias', 0);
sg('new_classifier', 'LIGHT');

sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', cache, from_order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
%x=sg('get_subkernel_weights');
%
%sg(sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, 0, mkl_stepsize, block, single_degree) );
%sg('set_subkernel_weights',x(1:order));
%
%%kmu=sg('get_kernel_matrix', 'TRAIN');
%
%sg(sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
%sg('set_subkernel_weights',x(1:order));
%%km=sg('get_kernel_matrix', 'TRAIN');

%sg('new_classifier LIGHT');
sg('c',C);
tic;
sg('svm_train');
tim_lo=toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'DNA');
out_ref=sg('svm_classify');
%prc_ref=calcrfcscore(out_ref, testlab);
%roc_ref=calcrocscore(out_ref, testlab);


traindat(traindat=='A')=0;
traindat(traindat=='C')=1;
traindat(traindat=='G')=2;
traindat(traindat=='T')=3;
traindat=uint8(traindat);

testdat(testdat=='A')=0;
testdat(testdat=='C')=1;
testdat(testdat=='G')=2;
testdat(testdat=='T')=3;
testdat=uint8(testdat);

sg('set_features', 'TRAIN', traindat', 'RAWDNA');
sg('set_labels', 'TRAIN', trainlab);
sg('c',C);
sg('svm_epsilon', epsilon);
sg('new_classifier','WDSVMOCAS',order, from_order);
tic;
sg('svm_train');
tim_lo=toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'RAWDNA');
out=sg('svm_classify');
%prc=calcrfcscore(out, testlab);
%roc=calcrocscore(out, testlab);

sg('set_features', 'TRAIN', traindat, 'RAWDNA', 'WD', order, from_order);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', testdat, 'RAWDNA', 'WD', order, from_order);
out_wdocas=sg('classify');

max(abs(out-out_ref))
max(abs(out_wdocas-out_ref))
max(abs(out_wdocas-out))

dat=[];
weights=sqrt((degree:-1:1)/sum(degree:-1:1))/4.281744;

N = size(traindat,1);
nDim = 0;
for d = 1:degree,
	nDim = nDim + 4^d;
end
nDim = nDim*N;

for j=1:size(traindat,2),
	dat(:,j)= zeros(nDim,1);
	offset = 0;
	for i=1:N,
	   val = 0;
	   for d = 1:degree
		   if i+d-1<=N,
			 val = 4*val + double(traindat(i+d-1,j));
			 dat(offset+val+1,j) = weights(d);
			 offset = offset + 4^d;
		   end
	   end
	end
end

traindat=sparse(dat);
testdat=traindat;

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', traindat);
out_ocas=sg('classify');

sg('set_features', 'TRAIN', dat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', dat);
out_docas=sg('classify');
max(abs(out-out_ocas))
max(abs(out-out_ref))
max(abs(out_ocas-out_ref))
max(abs(out_ocas-out_docas))

sg('set_features', 'TRAIN', [traindat;2*traindat]);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', [traindat;2*traindat]);
out1=sg('classify');

sg('clean_features','TRAIN');
sg('clean_features','TEST');
sg('add_dotfeatures', 'TRAIN', traindat);
sg('add_dotfeatures', 'TRAIN', 2*dat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('add_dotfeatures', 'TEST', traindat);
sg('add_dotfeatures', 'TEST', 2*dat);
out2=sg('classify');

max(abs(out1-out2))
