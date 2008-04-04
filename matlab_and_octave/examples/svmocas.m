C=10;
epsilon=1e-3;

rand('state',17);
num=16;
dim=10;
dist=0.001;
traindat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
scale=(dim*mean(traindat(:)));
traindat=sparse(traindat/scale);
trainlab=[-ones(1,num/2), +ones(1,num/2) ];


sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', 'svm_use_bias 0');
sg('send_command', 'svm_bufsize 1000');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', 'new_classifier SVMOCAS');
tic;
sg('send_command', 'train_classifier');
timeocas=toc

[b,W]=sg('get_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

b
W'
obj=sum(W.^2)+C*sum((1-trainlab.*(W'*traindat+b)).^2)
