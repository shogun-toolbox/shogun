addpath('../tools')
rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;
num_test=5000;
num_a=3;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
max_order=8;
order=20;
shift=15;
max_mismatch=0;
cache=10;
single_degree=-1;
x=shift*ones(1,len);
%x(:)=0;
shifts = int32(x(end:-1:1));


%generate some toy data
acgt='CCGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
aas=floor((shift+1)*rand(num_train,1));
idx=find(trainlab==1);
for i=1:length(idx),
	traindat(aa+aas(i),idx(i))='A';
end
aas=floor((shift+1)*rand(num_train,num_a));
idx=find(trainlab==-1);
for i=1:length(idx)/2,
	for j=1:num_a,
		traindat(aa(1)+aas(i,j),idx(i))='A';
	end
end
for i=length(idx)/2+1:length(idx),
	traindat(aa(1:2)+aas(i,1),idx(i))='AA';
	traindat(aa(1)+1+aas(i,2),idx(i))='A';
end

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
aas=floor((shift+1)*rand(num_test,1));
idx=find(testlab==1);
for i=1:length(idx),
	testdat(aa+aas(i),idx(i))='A';
end
aas=floor((shift+1)*rand(num_test,num_a));
idx=find(testlab==-1);
for i=1:length(idx)/2,
	for j=1:num_a,
		testdat(aa(1)+aas(i,j),idx(i))='A';
	end
end
for i=length(idx)/2+1:length(idx),
	testdat(aa(1:2)+aas(i,1),idx(i))='AA';
	testdat(aa(1)+1+aas(i,2),idx(i))='A';
end

%traindat'
%input('key to continue')
%keyboard;

%train svm
sg('loglevel', 'INFO');
sg('use_linadd', true);
sg('use_batch_computation', true);
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10, order, max_mismatch, len, shifts);
%sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
sg('train_classifier');

%evaluate svm on train data
sg('set_features', 'TEST', traindat,'DNA');
sg('set_labels', 'TEST', trainlab);
out=sg('classify');
fprintf('accuracy: %f roc: %f                                                                                        \n', mean(sign(out)==trainlab), calcrocscore(out,trainlab))


%evaluate svm on test data
sg('set_features', 'TEST', testdat,'DNA');
sg('set_labels', 'TEST', testlab);
out=sg('classify');
fprintf('accuracy: %f roc: %f                                                                                        \n', mean(sign(out)==testlab), calcrocscore(out,testlab))

W=sg('get_WD_scoring', max_order);
x={};
X=zeros(max_order,len);
l=0;
for i=1:max_order,
	i
	L=l+4^i*len;
	x{i}=W((l+1):L);
	x{i}=reshape(x{i},[4^i,len]);
	l=L;
	X(i,:)=max(abs(x{i}),[],1);
end

figure(1)
clf
set(gca,'fontsize',16);
imagesc(X)
title('Toy Dataset with motif AAA', 'fontsize', 16);
ylabel('k-mer', 'fontsize', 16);
xlabel('Position in Sequence', 'fontsize', 16);

figure(2)
clf
set(gca,'fontsize',16);
imagesc(X)
title('Toy Dataset with motif AAA', 'fontsize', 16);
ylabel('k-mer', 'fontsize', 16);
xlabel('Position in Sequence', 'fontsize', 16);

figure(3)
clf
set(gca,'fontsize',16);
imagesc(x{1})
title('Toy Dataset with motif AAA','fontsize', 16);
xlabel('Position in Sequence','fontsize', 16)
ylabel('Nucleotide','fontsize', 16)
set(gca,'YTick',1:4);
set(gca,'YTickLabel', {'A','C','G','T'});

figure(4)
clf
set(gca,'fontsize',16);
imagesc(x{2})
title('Toy Dataset with motif AAA','fontsize', 16);
xlabel('Position in Sequence','fontsize', 16)
ylabel('Dinucleotides','fontsize', 16)
set(gca,'YTick',1:16)
set(gca,'YTickLabel',{'AA','AC','AG','AT','CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT'})
colorbar

figure(5)
clf
set(gca,'fontsize',16);
imagesc(x{3})
title('Toy Dataset with motif AAA','fontsize', 16);
xlabel('Position in Sequence','fontsize', 16)
ylabel('Tri-mers','fontsize', 16)
set(gca,'YTick',1:4:64)
ticks={};
acgt='ACGT';
for i=1:4,
	for j=1:4,
		for k=1%:4,
			s=[ acgt(i) acgt(j) acgt(k)];
			ticks={ticks{:}, s};
		end
	end
end
set(gca,'YTickLabel',ticks)
colorbar

figure(6)
clf
set(gca,'fontsize',16);
imagesc(x{6})
title('Toy Dataset with motif AAA','fontsize', 16);
xlabel('Position in Sequence','fontsize', 16)
ylabel('Hexamers','fontsize', 16)
colorbar
