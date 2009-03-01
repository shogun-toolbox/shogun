rand('state',17);
num=1000;
dim=2;

%sg('loglevel', 'ALL');
traindat=[rand(dim,num/2)-0.3, rand(dim,num/2)+0.3];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-0.3, rand(dim,num/2)+0.3];
testlab=[-ones(1,num/2), ones(1,num/2) ];

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('new_classifier',  'LDA');
%sg('new_classifier', 'PERCEPTRON');
sg('train_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

sg('set_features', 'TEST', testdat);
testout=sg('classify');
testerr=mean(testlab~=sign(testout))

