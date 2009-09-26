rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=18;
num_test=42;
num_a=5;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Spectrum kernel parameters
order=8;
cache=10;
use_sign=0;
normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL

%generate some toy data
acgt='ACGT';
shift=40;
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=ones(1,num_test);

%sg('loglevel', 'ALL');

%%% spec
weights=(order:-1:1);
weights=weights/sum(weights);
km=zeros(size(traindat,2));
for o=1:order,
	sg('set_features', 'TRAIN', traindat, 'DNA');
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', o, order-1);
	sg('add_preproc', 'SORTWORDSTRING');
	sg('attach_preproc', 'TRAIN');
	sg('set_kernel', 'COMMSTRING', 'WORD', cache, use_sign, normalization);
	km=km+weights(o)*sg('get_kernel_matrix', 'TRAIN');
end

%%% wdspec
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD',cache, use_sign, normalization);

wkm=sg('get_kernel_matrix', 'TRAIN');

km(1)
wkm(1)
idx=order:size(traindat,1);
sum(traindat(idx,1)=='A')^2 + sum(traindat(idx,1)=='C')^2 + sum(traindat(idx,1)=='G')^2 + sum(traindat(idx,1)=='T')^2

%%% svm linadd off
sg('use_linadd',  0);
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', cache, use_sign, normalization);

sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');

sg('set_features', 'TEST', testdat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', testlab);
out=sg('classify');

%%% svm linadd on
sg('use_linadd', 1);
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', cache, use_sign, normalization);

sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');
sg('init_kernel_optimization');

sg('set_features', 'TEST', testdat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', testlab);
out_linadd=sg('classify');

max(abs(out_linadd-out))
