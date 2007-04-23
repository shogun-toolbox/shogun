rand('seed',17);

%sequence lengths, number of sequences
len=8;
num_train=100;
num_test=100;

%SVM regularization factor C
C=1;

%Spectrum kernel parameters
order=3;
cache=10;
use_sign=0;
normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL

%generate some toy data
acgt='ACGT';
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
idx=find(trainlab==1);
traindat(1,idx)='A';
traindat(2,idx)='C';
traindat(3,idx)='G';
traindat(4,idx)='T';
traindat(5,idx)='C';


traindat'
input('key to continue')

%train svm
sg('send_command', 'use_linadd 1' );
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'add_preproc SORTWORDSTRING') ;
sg('send_command', 'attach_preproc TRAIN') ;
sg('send_command', sprintf('set_kernel COMMSTRING WORD %d %d %s',cache, use_sign, normalization));

sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
sg('send_command', 'svm_train');
sg('send_command', 'init_kernel_optimization');

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'attach_preproc TEST') ;
sg('set_labels', 'TEST', trainlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))


consensus=sg('get_SPEC_consensus');
[b,alphas]=sg('get_svm');
sg('send_command', 'delete_kernel_optimization');
sg('set_features', 'TEST', [consensus traindat(:,1) traindat(:,end-1) traindat(:,end)], 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'attach_preproc TEST') ;
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
consensus'
out-b

kmers=ones(len, 4^len);
for i=2:(4^len),
	idx=len;
	kmers(:,i)=kmers(:,i-1);
	kmers(idx,i)=kmers(idx,i)+1;
	for j=1:len,
		if kmers(idx,i)>4,
			kmers(idx,i)=1;
			idx=idx-1;
			kmers(idx,i)=kmers(idx,i)+1;
		end
	end
end
kmers=acgt(kmers);

sg('set_features', 'TEST', kmers, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'attach_preproc TEST') ;
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
[b,alphas]=sg('get_svm');
out=out-b;
[v,i]=max(out);
v
i
