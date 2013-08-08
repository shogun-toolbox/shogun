modshogun

% Explicit examples on how to use clustering

addpath('tools');
fm_train=load_matrix('../data/fm_train_real.dat');

% KMeans
disp('KMeans')

k=4;
feats_train=RealFeatures(fm_train);
distance=EuclideanDistance(feats_train, feats_train);

kmeans=KMeans(k, distance);
kmeans.train();

c=kmeans.get_cluster_centers();
r=kmeans.get_radiuses();
