%rand('state',sum(100*clock));
rand('state',12345);
for i=1:1000,
num=ceil(1500*rand);
dims=ceil(1000*rand);
%sg('loglevel', 'ALL');
dist=rand;
traindat=[rand(dims,num)-dist rand(dims,num)+dist];
trainlab=sin(sum(traindat,1));
p=randperm(length(trainlab));
traindat=traindat(:,p);
trainlab=trainlab(:,p);
testdat=[rand(dims,num)-dist rand(dims,num)+dist];
testlab=sin(sum(testdat,1));
p=randperm(length(testlab));
testdat=testdat(:,p);
testlab=testlab(:,p);

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 10);
sg('new_regression', 'LIBSVR');
sg('c', 2);
sg('svr_tube_epsilon', 0.1);
tic;
sg('train_regression');
time_libsvm(i)=toc
[b2, alphas2]=sg('get_svm');
o2=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out2=sg('classify');
valerr2=mean(testlab~=sign(out2));

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 10);
sg('new_regression', 'SVRLIGHT');
sg('c', 2);
tic;
sg('train_regression');
time_light(i)=toc
[b3, alphas3]=sg('get_svm');
o3=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out3=sg('classify');
valerr3=mean(testlab~=sign(out3));

errs23(i)=max(abs(out2-out3))

obj23(i)=abs(o2-o3)
valerr3

if abs(o2-o3)>1e-5
	disp('obj error')
	keyboard
end

if (max(abs(out2-out3)) > 1e-4 )
	disp('out error')
	keyboard
end
end
