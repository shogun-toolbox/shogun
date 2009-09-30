init_shogun

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% knn
disp('KNN')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
distance=EuclidianDistance();

k=3;
num_threads=1;
labels=Labels(label_train_twoclass);

knn=KNN(k, distance, labels);
knn.parallel.set_num_threads(num_threads);
knn.train();

distance.init(feats_train, feats_test);
knn.classify().get_labels();
