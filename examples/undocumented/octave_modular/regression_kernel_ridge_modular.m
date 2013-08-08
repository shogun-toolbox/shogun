modshogun

addpath('tools');
label_train=load_matrix('../data/label_train_twoclass.dat');
fm_train=load_matrix('../data/fm_train_real.dat');
fm_test=load_matrix('../data/fm_test_real.dat');

% kernel ridge regression
disp('KRR')

feats_train=RealFeatures(fm_train);
feats_test=RealFeatures(fm_test);
width=0.8;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.9;
tau=1e-6;
num_threads=1;
labels=RegressionLabels(label_train);

krr=KernelRidgeRegression(tau, kernel, labels);
krr.parallel.set_num_threads(num_threads);
krr.train();

kernel.init(feats_train, feats_test);
out=krr.apply().get_labels();
