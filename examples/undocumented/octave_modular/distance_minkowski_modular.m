modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% minkowski metric
disp('MinkowskiMetric')

k=3;

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=MinkowskiMetric(feats_train, feats_train, k);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();
