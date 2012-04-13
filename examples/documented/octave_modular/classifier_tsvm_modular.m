%  V. Sindhwani, S.S. Keerthi. Newton Methods for Fast Solution of Semi-supervised
%  Linear SVMs. Large Scale Kernel Machines MIT Press (Book Chapter), 2007

init_shogun

addpath('tools');
%[labels,features]=libsvmread('../data/train_transduction.dat');
%labels=labels';
%features=double(features');
% Tsvm-SVMlin
disp('TSVM');
features=zeros(5,6);
for i=1:6,
	for j=1:5,
		features(j,i)=i*5+j;
	end
end
labels=[1;-1;0;0;0;1];
labels=double(labels');
realfeat=RealFeatures(features);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);

C=1000;
epsilon=1e-6;
num_threads=1;
labels=double(labels);
labels=Labels(labels);

svm=TSVM(C, feats_train, labels);
svm.io.set_loglevel(MSG_DEBUG);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(true);
disp('reached');
svm.train();
svm.get_bias()
svm.get_w();
