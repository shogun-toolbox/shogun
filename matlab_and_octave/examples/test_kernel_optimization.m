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
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel LINEAR CHAR 100');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', 'c 5');
sg('send_command', 'svm_train');
[b, alphas]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
valerr=mean(testlab~=sign(out));

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel LINEAR CHAR 100');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', 'c 5');
sg('send_command', 'svm_train');
[b2, alphas2]=sg('get_svm');
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel_optimization');
sg('send_command', 'init_kernel TEST');
out2=sg('svm_classify');
sg('send_command', 'delete_kernel_optimization');
valerr2=mean(testlab~=sign(out2));
errs(i)=max(abs(out-out2))
if (max(abs(out-out2)) > 1e-6)
	disp error
asdfasdfasdf
end
end
