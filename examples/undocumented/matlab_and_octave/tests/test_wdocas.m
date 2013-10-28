C=1
degree=1;
epsilon=1e-5;

x=[zeros(1,40000), ones(1,40000), 2*ones(1,40000), 3*ones(1,40000)];
y=[-ones(1,200), ones(1,200)];
x=x(randperm(length(x)));
y=y(randperm(length(y)));

traindat=uint8(reshape(x,400,400));
trainlab=y;
testdat=traindat;

%sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat, 'RAWDNA');
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'WDSVMOCAS', degree, degree);
sg('train_classifier');
sg('set_features', 'TEST', testdat, 'RAWDNA');
out=sg('classify');

dat=[];
weights=sqrt((degree:-1:1)/sum(degree:-1:1));

N = size(traindat,1);
nDim = 0;
for d = 1:degree,
	nDim = nDim + 4^d;
end
nDim = nDim*N;

for j=1:size(traindat,2),
	dat(:,j)= zeros(nDim,1);
	offset = 0;
	for i=1:N,
	   val = 0;
	   for d = 1:degree
		   if i+d-1<=N,
			 val = 4*val + double(traindat(i+d-1,j));
			 dat(offset+val+1,j) = weights(d);
			 offset = offset + 4^d;
		   end
	   end
	end
end

traindat=sparse(dat);
testdat=traindat;

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');
sg('set_features', 'TEST', testdat);
out_ref=sg('classify');
max(abs(out-out_ref))
