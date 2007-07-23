rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=10;
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

sg('send_command', 'loglevel ALL');

%%% spec
weights=(order:-1:1);
weights=weights/sum(weights);
km=zeros(size(traindat,2));
for o=1:order,
	sg('set_features', 'TRAIN', traindat, 'DNA');
	sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %i %i', o, order-1));
	sg('send_command', 'add_preproc SORTWORDSTRING') ;
	sg('send_command', 'attach_preproc TRAIN') ;
	sg('send_command', sprintf('set_kernel COMMSTRING WORD %d %d %s',cache, use_sign, normalization));
	sg('send_command', 'init_kernel TRAIN');
	km=km+weights(o)*sg('get_kernel_matrix');
end

%%% wdspec
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %i %i 0 r', order, order-1));
sg('send_command', 'add_preproc SORTWORDSTRING') ;
sg('send_command', 'attach_preproc TRAIN') ;
sg('send_command', sprintf('set_kernel WEIGHTEDCOMMSTRING WORD %d %d %s',cache, use_sign, normalization));
sg('send_command', 'init_kernel TRAIN');

wkm=sg('get_kernel_matrix');


max(abs(wkm(:)-km(:)))
