modshogun

addpath('tools');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% knn
disp('KNN')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
distance=EuclideanDistance(feats_train, feats_train);

k=3;
num_threads=1;
labels=MulticlassLabels(label_train_multiclass);

knn=KNN(k, distance, labels);
knn.parallel.set_num_threads(num_threads);
knn.train();

output=knn.apply(feats_test).get_labels();
