rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=100;
num_a=15;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

cache=10;

%generate some toy data
acgt='ARNDCQEGHILKMFPSTWYV';
shift=40;
rand('state',1);
traindat=acgt(ceil(20*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=min(floor((shift+1)*rand(num_train,1)), len-max(aa)-1);
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end

%%% local alignment kernel
sg('set_features', 'TRAIN', traindat, 'PROTEIN');
sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', cache);
km=sg('get_kernel_matrix', 'TRAIN');

imagesc(km)
