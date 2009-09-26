%rand('state',sum(100*clock));
rand('state',123455);
for i=1:1000,
%sg('loglevel', 'ALL');
num=ceil(1000*rand);
dims=ceil(100*rand);
dist=rand;
traindat=[rand(dims,num)-dist rand(dims,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];
p=randperm(length(trainlab));
traindat=traindat(:,p);
trainlab=trainlab(p);
testdat=[rand(dims,num)-dist rand(dims,num)+dist];
testlab=[-ones(1,num) ones(1,num)];
p=randperm(length(testlab));
testdat=testdat(:,p);
testlab=testlab(p);

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100);
sg('new_classifier', 'GPBTSVM');
sg('svm_epsilon', 1e-6);
sg('c', 2);
tic;
sg('train_classifier');
time_gpbt(i)=toc
[b, alphas]=sg('get_svm');
o=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out=sg('classify');
valerr=mean(testlab~=sign(out));

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100);
sg('new_classifier', 'LIBSVM');
sg('svm_epsilon', 1e-6);
sg('c', 2);
tic
sg('train_classifier');
time_libsvm(i)=toc
[b2, alphas2]=sg('get_svm');
o2=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out2=sg('classify');
valerr2=mean(testlab~=sign(out2));

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100);
sg('new_classifier', 'SVMLIGHT');
sg('svm_epsilon', 1e-6);
sg('c', 2);
tic;
sg('train_classifier');
time_light(i)=toc
[b3, alphas3]=sg('get_svm');
o3=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);

out3=sg('classify');
valerr3=mean(testlab~=sign(out3));

errs12(i)=max(abs(out-out2))
errs13(i)=max(abs(out-out3))
errs23(i)=max(abs(out2-out3))

obj12(i)=abs(o-o2)
obj13(i)=abs(o-o3)
obj23(i)=abs(o2-o3)
valerr

if abs(o-o2)>1e-4 | abs(o-o3)>1e-4 | abs(o2-o3)>1e-4
	disp('obj error')
	keyboard
end

if (max(abs(out-out2)) > 1e-4 || ...
		max(abs(out-out3)) > 1e-4 || ...
		max(abs(out2-out3)) > 1e-4 )
	disp('out error')
	keyboard
end
end
