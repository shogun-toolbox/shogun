num=1000;
dim=100;

sg('send_command', 'loglevel ALL');
rand('state',sum(100*clock));
traindat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
testlab=[-ones(1,num/2), ones(1,num/2) ];


sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_distance MINKOWSKI REAL 3.0');
sg('send_command', 'init_distance TRAIN');
sg('send_command', 'new_classifier KNN');
sg('send_command', 'train_classifier 2');

sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out4=sg('classify');
valerr4=mean(testlab~=sign(out4))
sg('set_features', 'TEST', traindat);
sg('set_labels', 'TEST', trainlab);
outt4=sg('classify');
valerrt4=mean(trainlab~=sign(outt4))

