C=100000;
epsilon=1e-3;

rand('state',17);
num=10000;
dim=20;
dist=0.08;

traindat=sparse([rand(dim,num/2)-dist, rand(dim,num/2)+dist]);
trainlab=[-ones(1,num/2), ones(1,num/2) ];

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', 'new_classifier LPM');
tic;
sg('send_command', 'train_classifier');
timelpm=toc

[b,W]=sg('get_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

b
W'
