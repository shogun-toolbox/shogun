% Explicit examples on how to use clustering

addpath('tools');
label_train=load_matrix('../data/label_train_oneclass.dat');
fm_train=load_matrix('../data/fm_train_real.dat');


% KMEANS
disp('KMeans');

k=3;
iter=1000;

sg('set_features', 'TRAIN', fm_train);
sg('set_labels', 'TRAIN', label_train);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('new_clustering', 'KMEANS');

sg('init_distance', 'TRAIN');
sg('train_clustering', k, iter);
[radi, centers]=sg('get_clustering');


% Hierarchical
disp('Hierarchical');

merges=3;

sg('set_features', 'TRAIN', fm_train);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('new_clustering', 'HIERARCHICAL');

sg('init_distance', 'TRAIN');
sg('train_clustering', merges);
[merge_distance, pairs]=sg('get_clustering');
