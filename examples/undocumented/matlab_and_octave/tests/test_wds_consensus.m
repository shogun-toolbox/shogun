rand('seed',17);
%sequence lengths, number of sequences
len=8;
num_train=1000;

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
order=3;
shift=15;
max_mismatch=0;
cache=10;
single_degree=-1;
x=shift*ones(1,len);
x(:)=0;
shifts = int32(x(end:-1:1));


%generate some toy data
acgt='ACGT';
traindat=[acgt(ceil(4*rand(len,num_train/2))) acgt(ceil(4*rand(len,num_train/2)))];
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
traindat'
input('key to continue')

%train svm
%sg('loglevel', 'INFO' );
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10', order, max_mismatch, len, shifts);
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');
consensus=sg('get_WD_consensus');
consensus'

c=traindat(:,trainlab==1);
c(c=='A')=1;
c(c=='C')=2;
c(c=='G')=3;
c(c=='T')=4;
simpleconsensus=acgt(floor(median(c')))';
simpleconsensus'

sg('set_features', 'TEST', [ consensus simpleconsensus' traindat(:,end-20)' traindat(:,end)'], 'DNA');
out=sg('classify');
[b,alphas]=sg('get_svm');
sprintf('%5f\n', out'-b)

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
out=sg('classify');
[b,alphas]=sg('get_svm');
out=out-b;
[v,i]=max(out);
v
i
