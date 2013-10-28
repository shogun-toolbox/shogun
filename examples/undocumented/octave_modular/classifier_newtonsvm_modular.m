modshogun

addpath('tools');
label=load_matrix('../data/label_train_dna.dat');
data=load_matrix('../data/fm_train_dna.dat');
fm_test_real=load_matrix('../data/fm_test_dna.dat');

% Newton SVM

disp('NewtonSVM')
data=double(data);
%fm_test_real=double(fm_test_real);
realfeat=RealFeatures(data);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
%realfeat=RealFeatures(fm_test_real);
%feats_test=SparseRealFeatures();
%feats_test.obtain_from_simple(realfeat);

C=1.0;
epsilon=1e-5;
num_threads=1;
label=double(label);
labels=BinaryLabels(label);

svm=NewtonSVM(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(true);
svm.train();
%svm.set_features(feats_test);
svm.get_bias();
svm.get_w();
%svm.apply().get_labels();
