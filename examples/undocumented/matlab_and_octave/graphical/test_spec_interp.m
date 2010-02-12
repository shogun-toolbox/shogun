rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_a=5;
shift=20;
cache=10;
use_sign=false;
normalization='NO'; %NO,SQRT,LEN,SQLEN,FULL
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
order=4;   % of spectrum kernel
max_order=4;  % of POIMs

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
sg('use_linadd', true);
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('add_preproc', 'SORTWORDSTRING') ;
sg('attach_preproc', 'TRAIN') ;
sg('set_kernel', 'COMMSTRING', 'WORD', cache, use_sign, normalization);
sg('set_kernel_normalization', 'IDENTITY');

sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');
[b,alphas]=sg('get_svm');
%b=0;
sg('init_kernel_optimization');

normal=sg('get_kernel_optimization');

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', trainlab);
out=sg('classify');
fprintf('accuracy: %f                                                                                         \n', mean(sign(out)==trainlab))

%evaluate svm on train data
sg('set_features', 'TEST', traindat, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('attach_preproc', 'TEST');
sg('set_labels', 'TEST', trainlab);
out=sg('classify');
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
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
	sg('attach_preproc', 'TEST');
	out=sg('classify');
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
	x{i} = x{i} - mean(x{i});
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


for i=1:max_order,
	figure(200+i);
	t1= x{i};
	t2 = xx{i}';
	[m1,i1] = max( t1 );
	[m2,i2] = max( t2 );
	assert( i1 == i2 );
	t1(i1) = [];
	t2(i2) = [];
	plot( t1, t2, 'LineStyle', 'none', 'Marker', 'x', 'LineWidth', 2, 'MarkerSize', 5 );
	grid on;
	fprintf( 'scaling %.2f\n', m1/m2 );
end


