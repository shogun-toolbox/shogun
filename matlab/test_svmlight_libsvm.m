for i=1:1000,
rand('state',sum(100*clock));
traindat=rand(100,1000);
trainlab=2*round(rand(1,1000))-1;
testdat=rand(100,1000);
testlab=2*round(rand(1,1000))-1;
gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', 'set_kernel GAUSSIAN REAL 100');
gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'new_svm LIGHT');
gf('send_command', 'c 5');
gf('send_command', 'svm_train');
[b, alphas]=gf('get_svm');
gf('set_features', 'TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
out=gf('svm_classify');
valerr=mean(testlab~=sign(out));

gf('set_features', 'TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', 'set_kernel GAUSSIAN REAL 100');
gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'new_svm LIBSVM');
gf('send_command', 'c 5');
gf('send_command', 'svm_train');
[b2, alphas2]=gf('get_svm');
gf('set_features', 'TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
out2=gf('svm_classify');
valerr2=mean(testlab~=sign(out2));
errs(i)=max(abs(out-out2))
if (max(abs(out-out2)) > 1e-6)
	disp error
asdfasdfasdf
end
end
