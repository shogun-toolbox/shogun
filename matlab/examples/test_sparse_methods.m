rand('state',17);
num=20;
dim=10;
dist=0.3;
C=10;

sg('send_command', 'loglevel ALL');
traindat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
testlab=[-ones(1,num/2), ones(1,num/2) ];

sg('set_features', 'TRAIN', traindat);
sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;

sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
%sg('send_command', 'new_classifier SVMLIN');
sg('send_command', 'new_classifier SUBGRADIENTSVM');
sg('send_command', 'train_classifier');

sg('set_features', 'TEST', traindat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

sg('set_features', 'TEST', testdat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
testout=sg('classify');
testerr=mean(testlab~=sign(testout))
