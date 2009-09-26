rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_test=5000;
num_a=5;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Spectrum kernel parameters
order=5;
cache=10;
use_sign=true;
normalization='FULL'; %NO,SQRT,LEN,SQLEN,FULL

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
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
aas=floor((shift+1)*rand(num_test,1));
idx=find(testlab==1);
for i=1:length(idx),
	testdat(aa+aas(i),idx(i))='A';
end

%traindat'
%input('key to continue')

%train svm
sg('use_linadd', true);
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'COMMSTRING', 'WORD', cache, use_sign, normalization);

sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');
sg('init_kernel_optimization');

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', trainlab);
out=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', testlab);
out=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==testlab))
