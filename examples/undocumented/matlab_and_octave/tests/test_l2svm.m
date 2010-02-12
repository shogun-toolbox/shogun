C=1;
epsilon=1e-3;

randn('state',17);
num=10000;
dim=200;
dist=0.1;
use_bias=1;

L=load('real-sim.mat');
traindat=L.X';
trainlab=L.Y';
num=size(traindat,2);
dim=size(traindat,1);

%traindat=sparse([randn(dim,num/2)-dist, randn(dim,num/2)+dist]);
%trainlab=[ones(1,num/2), -ones(1,num/2) ];

%sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C;
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('new_classifier', 'LIBLINEAR_L2');
tic;
sg('train_classifier');
time_liblinear=toc

[b1,W1]=sg('get_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout));

b1
W1';
obj1=sum(W1.^2)+C*sum(max(1-trainlab.*(W1'*traindat-b1)).^2)

%sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', use_bias);
sg('new_classifier', 'SVMLIN');
tic;
sg('train_classifier');
time_svmlin=toc
%
[b2,W2]=sg('get_classifier');
%
sg('set_features', 'TEST', traindat);
trainout2=sg('classify');
trainerr2=mean(trainlab~=sign(trainout2));
%
b2
W2';
obj2=sum(W2.^2)+C*sum(max(1-trainlab.*(W2'*traindat+b2)).^2)

%trainout(1:10);
%trainout2(1:10);
%
%%sg('loglevel', 'ERROR');
%%sg('set_features', 'TRAIN', traindat);
%%sg('set_labels', 'TRAIN', trainlab+1/2);
%%sg('c', C);
%%sg('svm_use_bias', 1);
%%sg('new_classifier', 'GMNPSVM');
%%sg('set_kernel', 'LINEAR', 'SPARSEREAL', 10, 1.0);
%%tic;
%%sg('train_classifier');
%%timeliblinear=toc
%%
%%%[b3,W3]=sg('get_classifier');
%%
%%sg('set_features', 'TEST', traindat);
%%trainout3=sg('classify')*2-1;
%%trainerr3=mean(trainlab~=sign(trainout3))
%sg('loglevel', 'ERROR');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_use_bias', 1);
sg('svm_qpsize', 500);
sg('new_classifier', 'GPBTSVM');
sg('set_kernel', 'LINEAR', 'SPARSEREAL', 200, 1.0);
tic;
sg('train_classifier');
time_gpbt=toc
[b_gpbt,W_gpbt]=sg('get_classifier');

%sg('loglevel', 'ERROR');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_use_bias', 1);
sg('svm_qpsize', 42);
sg('new_classifier', 'SVMLIGHT');
sg('set_kernel', 'LINEAR SPARSEREAL', 200, 1.0);
tic;
sg('train_classifier');
time_light=toc
[b_light,W_light]=sg('get_classifier');

%sg('loglevel', 'ERROR');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('c', C);
sg('svm_use_bias', 1);
sg('new_classifier', 'LIBSVM');
sg('set_kernel', 'LINEAR', 'SPARSEREAL', 10, 1.0);
tic;
sg('train_classifier');
time_libsvm=toc
[b_libsvm,W_libsvm]=sg('get_classifier');



%addpath ../../other_ml_suites/liblinear-1.1/matlab
%model=train(trainlab',traindat', sprintf('-s 2 -c %f -e %g -B %d', C, epsilon, use_bias));
%%trainerr1=mean(trainlab~=sign(o1'))
%if (length(model.w)>dim)
%	b1_ref=model.w(end);
%	W1_ref=model.w(1:end-1);
%else
%	b1_ref=0;
%	W1_ref=model.w(:);
%end
%b1_ref
%W1_ref';
%obj1_ref=sum(W1_ref.^2)+C*sum(max(1-trainlab.*(W1_ref'*traindat-b1_ref)).^2)
%
%addpath ../../other_ml_suites/svmlin-v1.0
%[w2_ref,o1]=svmlin(sprintf('-A 1 -W %f', 1/(2*C)),traindat',trainlab');
%trainerr1=mean(trainlab~=sign(o1'));
%w2_ref=w2_ref*trainlab(1);
%b2_ref=w2_ref(end);
%W2_ref=w2_ref(1:end-1);
%b2_ref
%W2_ref';
%obj2_ref=sum(W2_ref.^2)+C*sum(max(1-trainlab.*(W2_ref'*traindat+b2_ref)).^2)
%
%fprintf('obj1: %10.10f (diff:%10.10f)\n', obj1, obj1-obj1_ref)
%fprintf('obj2: %10.10f (diff:%10.10f)\n', obj2, obj2-obj2_ref)
%fprintf('diff: %10.10f\n', abs(obj1-obj2)/min(obj1,obj2))
