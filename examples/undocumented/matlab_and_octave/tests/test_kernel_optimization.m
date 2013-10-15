alphab=[ 'A', 'C', 'G', 'T' ];
for i=1:100
rand('state',sum(100*clock));

for j=1:100,
	for k=1:1000,
	 traindat(j,k)=char(alphab(round(3*rand)+1));
	 testdat(j,k)=char(alphab(round(3*rand)+1));
	end
end

%traindat=rand(100,1000);
trainlab=2*round(rand(1,1000))-1;
%testdat=rand(100,1000);
testlab=2*round(rand(1,1000))-1;
%sg('loglevel', 'INFO');
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'LINEAR', 'CHAR', 100);
sg('new_classifier', 'SVMLIGHT');
sg('c', 5);
sg('train_classifier');
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
out=sg('classify');
valerr=mean(testlab~=sign(out));

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'LINEAR', 'CHAR', 100);
sg('new_classifier', 'SVMLIGHT');
sg('c', 5);
sg('train_classifier');
[b2, alphas2]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
sg('init_kernel_optimization');
out2=sg('classify');
sg('delete_kernel_optimization');
valerr2=mean(testlab~=sign(out2));
errs(i)=max(abs(out-out2))
if (max(abs(out-out2)) > 1e-6)
end
end
