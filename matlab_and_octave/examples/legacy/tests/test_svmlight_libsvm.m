%rand('state',sum(100*clock));
rand('state',123455);
for i=1:1000,
sg('send_command', 'loglevel ALL');
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
sg('send_command', 'set_kernel GAUSSIAN REAL 100');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm GPBTSVM');
sg('send_command','svm_epsilon 1e-6')
sg('send_command', 'c 2');
tic;
sg('send_command', 'svm_train');
time_gpbt(i)=toc
[b, alphas]=sg('get_svm');
o=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
valerr=mean(testlab~=sign(out));

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel GAUSSIAN REAL 100');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIBSVM');
sg('send_command','svm_epsilon 1e-6')
sg('send_command', 'c 2');
tic
sg('send_command', 'svm_train');
time_libsvm(i)=toc
[b2, alphas2]=sg('get_svm');
o2=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out2=sg('svm_classify');
valerr2=mean(testlab~=sign(out2));

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel GAUSSIAN REAL 100');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command','svm_epsilon 1e-6')
sg('send_command', 'c 2');
tic;
sg('send_command', 'svm_train');
time_light(i)=toc
[b3, alphas3]=sg('get_svm');
o3=sg('get_svm_objective');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');

out3=sg('svm_classify');
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
