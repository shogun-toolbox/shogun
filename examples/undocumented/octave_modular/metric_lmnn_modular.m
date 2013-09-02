modshogun;

label_train_fname='../data/label_train_multiclass.dat';
train_fname='../data/fm_train_real.dat';
test_fname='../data/fm_test_real.dat';

% wrap data into Shogun objects
feats_train=RealFeatures(CSVFile(train_fname));
feats_test=RealFeatures(CSVFile(test_fname));
labels_train=MulticlassLabels(CSVFile(label_train_fname));

% distance learning with LMNN 
disp('LMNN')

% number of target neighbours per example
k=3;
lmnn=LMNN(feats_train,labels_train,k);
init_transform=eye(feats_train.get_num_features());
lmnn.train(init_transform);

% retrieve distance learnt by LMNN
lmnn_distance=lmnn.get_distance();

% perform multiclass classification using KNN with the distance learnt
knn=KNN(k,lmnn_distance,labels_train);
knn.train();
output=knn.apply(feats_test).get_labels();
