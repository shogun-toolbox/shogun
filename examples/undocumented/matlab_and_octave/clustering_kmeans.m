addpath('tools');
fm_train=load_matrix('../data/fm_train_real.dat');

% KMEANS
disp('KMeans');

k=3;
iter=1000;

sg('set_features', 'TRAIN', fm_train);
sg('set_distance', 'EUCLIDEAN', 'REAL');
sg('new_clustering', 'KMEANS');

sg('train_clustering', k, iter);
[radi, centers]=sg('get_clustering');
