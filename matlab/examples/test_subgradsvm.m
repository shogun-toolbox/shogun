C=10;
epsilon=1e-5;

load /home/sonne/vojtech/subgradsvm/uci_spambase.mat
data=[];
traindat=x';
trainlab=t';
testdat=x';
testlab=t';

%rand('state',17);
%num=20000;
%dim=10;
%dist=0.03;
%%num=20;
%%dim=1000;
%%dist=0.01;
%
%traindat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
%traindat=traindat/(dim*mean(traindat(:)));
%trainlab=[-ones(1,num/2), ones(1,num/2) ];
%
%testdat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
%testdat=testdat/(dim*mean(testdat(:)));;
%testlab=[-ones(1,num/2), ones(1,num/2) ];

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;

sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_epsilon %10.10f', epsilon));
%sg('send_command', 'new_classifier SVMLIN');
sg('send_command', 'new_classifier SUBGRADIENTSVM');
tic;
sg('send_command', 'train_classifier');
timesubgradsvm=toc

sg('set_features', 'TEST', traindat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))
[b,W]=sg('get_classifier');
F_subgrad = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout)),1 - trainout))

sg('set_features', 'TEST', testdat);
sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
testout=sg('classify');
testerr=mean(testlab~=sign(testout))

%%%%LIGHT%%%
sg('set_features', 'TRAIN', traindat);
%sg('send_command', 'use_linadd 0');
%sg('send_command', 'use_batch_computation 0');
%sg('send_command', 'svm_qpsize 10');
%sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
%sg('send_command', 'set_kernel LINEAR SPARSEREAL 1000 1.0');
sg('send_command', 'set_kernel LINEAR REAL 10 1.0');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', sprintf('svm_epsilon %10.10f', epsilon));
sg('send_command', 'new_classifier SVMLIGHT');
tic;
sg('send_command', 'train_classifier');
timelight=toc

sg('send_command', 'init_kernel_optimization');
sg('set_features', 'TEST', traindat);
%sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
sg('send_command', 'init_kernel TEST');
obj_light=sg('get_svm_objective')
trainout_reflight=sg('classify');
trainerr_reflight=mean(trainlab~=sign(trainout_reflight))

W=sg('get_kernel_optimization');
F_light = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout_reflight)),1 - trainout_reflight));
F_light

%%%%LIBSVM%%%
sg('set_features', 'TRAIN', traindat);
%sg('send_command', 'convert TRAIN SIMPLE REAL SPARSE REAL') ;
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', 'set_kernel LINEAR REAL 50 1.0');
%sg('send_command', 'set_kernel LINEAR SPARSEREAL 50 1.0');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', sprintf('svm_epsilon %10.10f', epsilon));
sg('send_command', 'new_classifier LIBSVM');
tic;
sg('send_command', 'train_classifier');
timelibsvm=toc

sg('send_command', 'init_kernel_optimization');
sg('set_features', 'TEST', traindat);
%sg('send_command', 'convert TEST SIMPLE REAL SPARSE REAL') ;
sg('send_command', 'init_kernel TEST');
obj_libsvm=sg('get_svm_objective')
trainout_reflibsvm=sg('classify');
trainerr_reflibsvm=mean(trainlab~=sign(trainout_reflibsvm))

max(abs(trainout-trainout_reflight)) 
max(abs(trainout-trainout_reflibsvm))
max(abs(trainout_reflibsvm-trainout_reflight))

timesubgradsvm
timelight
timelibsvm

W=sg('get_kernel_optimization');
F_libsvm = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout_reflibsvm)),1 - trainout_reflibsvm));
F_libsvm

[b,a]=sg('get_classifier');
alpha=zeros(size(traindat,2),1);
alpha(a(:,2)+1)=a(:,1);

%F_libsvm_alpha=0.5*alpha'*(traindat'*traindat)*alpha + C*sum(max(zeros(size(trainout_reflibsvm)),1 - trainout_reflibsvm))
disp('objectives')
fprintf('light:%f\n',F_light)
fprintf('libsvm:%f\n',F_libsvm)
fprintf('subgrad:%f\n',F_subgrad)
