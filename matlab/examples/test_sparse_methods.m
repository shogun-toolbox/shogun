rand('state',17);
num=200000;
dim=1000;
dist=0.01;
C=10;
epsilon=1e-5;

sg('send_command', 'loglevel ALL');
traindat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
traindat=traindat/(dim*mean(traindat(:)));
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
testdat=testdat/(dim*mean(testdat(:)));;
testlab=[-ones(1,num/2), ones(1,num/2) ];

sg('set_features', 'TRAIN', traindat);
sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;

sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_epsilon %f', epsilon));
%sg('send_command', 'new_classifier SVMLIN');
sg('send_command', 'new_classifier SUBGRADIENTSVM');
tic;
sg('send_command', 'train_classifier');
timesubgradsvm=toc

sg('set_features', 'TEST', traindat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

sg('set_features', 'TEST', testdat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
testout=sg('classify');
testerr=mean(testlab~=sign(testout))

%%%%LIGHT%%%
sg('set_features', 'TRAIN', traindat);
%sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
%sg('send_command', 'set_kernel LINEAR SPARSEREAL 1000 1.0');
sg('send_command', 'set_kernel LINEAR REAL 10 1.0');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_classifier SVMLIGHT');
tic;
sg('send_command', 'train_classifier');
timelight=toc

sg('send_command', 'init_kernel_optimization');
sg('set_features', 'TEST', traindat);
%sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
sg('send_command', 'init_kernel TEST');
trainout_reflight=sg('classify');
trainerr_reflight=mean(trainlab~=sign(trainout_reflight))

%%%LIBSVM%%%
sg('set_features', 'TRAIN', traindat);
sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', 'set_kernel LINEAR SPARSEREAL 500 1.0');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_classifier LIBSVM');
tic;
sg('send_command', 'train_classifier');
timelibsvm=toc

sg('send_command', 'init_kernel_optimization');
sg('set_features', 'TEST', traindat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
sg('send_command', 'init_kernel TEST');
trainout_reflibsvm=sg('classify');
trainerr_reflibsvm=mean(trainlab~=sign(trainout_reflibsvm))

max(abs(trainout-trainout_reflight)) 
max(abs(trainout-trainout_reflibsvm))
max(abs(trainout_reflibsvm-trainout_reflight))

timesubgradsvm
timelight
timelibsvm

