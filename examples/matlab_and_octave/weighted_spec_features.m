rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=10;
num_a=5;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));
epsilon=1e-8;

%SVM regularization factor C
C=1;

%Spectrum kernel parameters
order=8;
cache=10;
use_sign=false;
normalize=true;

if normalize,
	normalization='FULL'; %NO,SQRT,LEN,SQLEN,FULL
else
	normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL
end

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
	sg('set_kernel', 'COMMSTRING', 'WORD',cache, use_sign, "NO");
	km=km+weights(o)*sg('get_kernel_matrix', 'TRAIN');
end

km2=km;
if normalize,
	for i=1:size(km,1),
		for j=1:size(km,2),
			km2(i,j)=km(i,j)/(sqrt(km(i,i)*km(j,j)));
		end
	end
end

%%% wdspec
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', cache, use_sign, normalization);

feat=sg('get_features','TRAIN');
wkm=sg('get_kernel_matrix', 'TRAIN');


fprintf('max diff %g\n', max(abs(wkm(:)-km2(:))))

sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('use_linadd', true);
sg('new_classifier', 'SVMLIGHT');
sg('set_labels','TRAIN', trainlab);
sg('train_classifier');
[bias, alphas]=sg('get_classifier');
sg('init_kernel_optimization');
svmw=sg('get_kernel_optimization');
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, 0, 'r');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TEST');
out_ref=sg('classify');

sg('c', C);
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('use_linadd', false);
sg('new_classifier', 'SVMLIGHT');
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('set_labels','TRAIN', trainlab);
sg('set_kernel','CUSTOM', km2, 'FULL');
sg('train_classifier');
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
out_ref2=sg('classify');

traindat(traindat=='A')=0;
traindat(traindat=='C')=1;
traindat(traindat=='G')=2;
traindat(traindat=='T')=3;
traindat=uint8(traindat);
testdat=uint8(traindat);

clear sg
sg('svm_use_bias', 0);
sg('svm_epsilon', epsilon);
sg('set_labels','TRAIN', trainlab);
sg('set_features', 'TRAIN', traindat, 'RAWDNA','WSPEC', order, order-1, normalize);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
[bias_ocas, alphas_ocas]=sg('get_classifier');
sg('set_features', 'TEST', testdat, 'RAWDNA','WSPEC', order, order-1, normalize);
out=sg('classify');


fprintf('max out diff %g\n', max(abs(out-out_ref)))
fprintf('max out diff %g\n', max(abs(out-out_ref2)))

max(abs(svmw(1:length(alphas_ocas))-alphas_ocas'))

%o=[];
%for i=1:length(feat),
%	o(i)=alphas_ocas*feat{i};
%end
