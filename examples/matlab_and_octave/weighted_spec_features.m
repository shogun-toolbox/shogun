rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=10;
num_a=5;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));
epsilon=1e-6;

%SVM regularization factor C
C=1;

%Spectrum kernel parameters
order=8;
cache=10;
use_sign=false;
normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL
normalize=false;

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

sg('loglevel', 'ALL');

%%% spec
weights=(order:-1:1);
weights=weights/sum(weights);
km=zeros(size(traindat,2));
for o=1:order,
	sg('set_features', 'TRAIN', traindat, 'DNA');
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', o, order-1);
	sg('add_preproc', 'SORTWORDSTRING');
	sg('attach_preproc', 'TRAIN');
	sg('set_kernel', 'COMMSTRING', 'WORD',cache, use_sign, normalization);
	sg('init_kernel', 'TRAIN');
	km=km+weights(o)*sg('get_kernel_matrix');
end

%%% wdspec
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', cache, use_sign, normalization);
sg('init_kernel', 'TRAIN');

wkm=sg('get_kernel_matrix');


fprintf('max diff %g\n', max(abs(wkm(:)-km(:))))

sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMLIGHT');
sg('set_labels','TRAIN', trainlab);
sg('train_classifier');
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TEST');
out_ref=sg('classify');

traindat(traindat=='A')=0;
traindat(traindat=='C')=1;
traindat(traindat=='G')=2;
traindat(traindat=='T')=3;
traindat=uint8(traindat);

sg('set_features', 'TRAIN', traindat, 'RAWDNAWSPEC', order, order-1, normalize);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', traindat, 'RAWDNAWSPEC', order, order-1, normalize);
out=sg('classify');


fprintf('max out diff %g\n', max(abs(out-out_ref)))
