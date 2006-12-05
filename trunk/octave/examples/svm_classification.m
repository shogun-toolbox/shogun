num=ceil(5000*rand);
traindat=[rand(2,num)-0.2 rand(2,num)+0.2];
trainlab=[-ones(1,num) ones(1,num)];

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel GAUSSIAN REAL 10 0.1');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIBSVM');
sg('send_command','svm_epsilon 1e-5')
sg('send_command', 'c 10');
sg('send_command', 'svm_train');

mi=min(traindat');
ma=max(traindat');
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');

figure(1)
clf
out=reshape(out,50,50);
surf(out)
