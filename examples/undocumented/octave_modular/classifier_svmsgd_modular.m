modshogun

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% sgd
disp('SVMSGD')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.9;
num_iter=5
num_threads=1;
labels=BinaryLabels(label_train_twoclass);

svm=SVMSGD(C, feats_train, labels);
svm.set_epochs(num_iter)
%svm.io.set_loglevel(0);
svm.train();

svm.set_features(feats_test);
svm.apply().get_labels();
