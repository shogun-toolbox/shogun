rand('state',17);
num=1000;
dim=2;

sg('send_command', 'loglevel ALL');
traindat=[rand(dim,num/2)-0.3, rand(dim,num/2)+0.3];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-0.3, rand(dim,num/2)+0.3];
testlab=[-ones(1,num/2), ones(1,num/2) ];

sg('send_command', 'loglevel DEBUG');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'new_classifier LDA');
%sg('send_command', 'new_classifier PERCEPTRON');
sg('send_command', 'train_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

sg('set_features', 'TEST', testdat);
testout=sg('classify');
testerr=mean(testlab~=sign(testout))

