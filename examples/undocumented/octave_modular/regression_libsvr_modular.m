modshogun

addpath('tools');
label_train=load_matrix('../data/label_train_twoclass.dat');
fm_train=load_matrix('../data/fm_train_real.dat');
fm_test=load_matrix('../data/fm_test_real.dat');

%% libsvm based support vector regression
disp('LibSVR')

feats_train=RealFeatures(fm_train);
feats_test=RealFeatures(fm_test);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=1.2;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=3;
labels=RegressionLabels(label_train);

svr=LibSVR(C, epsilon, kernel, labels);
svr.set_tube_epsilon(tube_epsilon);
svr.parallel.set_num_threads(num_threads);
svr.train();

kernel.init(feats_train, feats_test);
out=svr.apply().get_labels();

