init_shogun

num=50; %number of example
len=10; %number of dimensions
dist=1.5;

% Explicit examples on how to use regressions

traindata_real=[randn(len,num)-dist, randn(len,num)+dist];
testdata_real=[randn(len,num+7)-dist, randn(len,num+7)+dist];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm-based
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% libsvm based support vector regression
disp('SVRLight')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=1;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svr=SVRLight(C, epsilon, kernel, labels);
svr.set_tube_epsilon(tube_epsilon);
svr.parallel.set_num_threads(num_threads);
svr.train();

kernel.init(feats_train, feats_test);
svr.classify().get_labels();

%% libsvm based support vector regression
disp('LibSVR')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=1;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svr=LibSVR(C, epsilon, kernel, labels);
svr.set_tube_epsilon(tube_epsilon);
svr.parallel.set_num_threads(num_threads);
svr.train();

kernel.init(feats_train, feats_test);
out=svr.classify().get_labels();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% misc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% kernel ridge regression
disp('KRR')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=0.8;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.42;
tau=1e-6;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

krr=KRR(tau, kernel, labels);
krr.parallel.set_num_threads(num_threads);
krr.train();

kernel.init(feats_train, feats_test);
out=krr.classify().get_labels();
