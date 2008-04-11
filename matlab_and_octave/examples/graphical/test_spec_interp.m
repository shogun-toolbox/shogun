rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_a=5;
shift=20;
cache=10;
use_sign=0;
normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
order=5;
max_order=5;

rand('state',1);
acgt='ACGT';
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
idx=find(trainlab==1);
for i=1:length(idx)
	traindat(1:3,idx(i))='AAA';
	%traindat(3:22,idx(i))='AAAAAAAAAAAAAAAAAAAA';
	traindat(24:31,idx(i))='CCCCCCCC';
end

%%generate some toy data
%acgt='CCGT';
%rand('state',1);
%traindat=acgt(ceil(4*rand(len,num_train)));
%trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
%aas=floor((shift+1)*rand(num_train,1));
%idx=find(trainlab==1);
%for i=1:length(idx),
%	traindat(aa+aas(i),idx(i))='A';
%end
%aas=floor((shift+1)*rand(num_train,num_a));
%idx=find(trainlab==-1);
%for i=1:length(idx)/2,
%	for j=1:num_a,
%		traindat(aa(1)+aas(i,j),idx(i))='A';
%	end
%end
%for i=length(idx)/2+1:length(idx),
%	traindat(aa(1:3)+aas(i,1),idx(i))='AAA';
%	traindat(aa(1)+1+aas(i,2),idx(i))='A';
%end

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
[b,alphas]=sg('get_svm');
%b=0;
sg('send_command', 'init_kernel_optimization');

normal=sg('get_kernel_optimization');

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'attach_preproc TEST') ;
sg('set_labels', 'TEST', trainlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i', order, order-1));
sg('send_command', 'attach_preproc TEST') ;
sg('set_labels', 'TEST', trainlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))

xx={};
for o=1:max_order,
	acgt='ACGT';
	ord=2*(order-1)+o;
	kmers=ones(ord, 4^ord);
	for i=2:(4^ord),
		idx=ord;
		kmers(:,i)=kmers(:,i-1);
		kmers(idx,i)=kmers(idx,i)+1;
		for j=1:ord,
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
	out=out-b;

	xx{o}=[];
	omers=ones(o, 4^o);
	for i=2:(4^o),
		idx=o;
		omers(:,i)=omers(:,i-1);
		omers(idx,i)=omers(idx,i)+1;
		for j=1:o,
			if omers(idx,i)>4,
				omers(idx,i)=1;
				idx=idx-1;
				omers(idx,i)=omers(idx,i)+1;
			end
		end
	end
	omers=acgt(omers);
	for s=omers,
		i=strmatch(s,kmers((order):(order+o-1),:)');
		x=mean(out(i))-mean(out);
		xx{o}=[xx{o} x];
		%fprintf('%s:%g\n', s, x)
	end
end

W=sg('get_SPEC_scoring', max_order);

x={};
X=zeros(max_order);
l=0;
for i=1:max_order,
	L=l+4^i;
	x{i}=W((l+1):L);
	l=L;
	X(i)=max(abs(x{i}));
end

%for i=1:max_order,
%	figure(i)
%	bar(x{i})
%end

for i=1:max_order,
	figure(100+i);
	foo=[x{i}, xx{i}'];
	bar(foo)
	%foo
	max(abs(foo(:,1)-foo(:,2)))
end
