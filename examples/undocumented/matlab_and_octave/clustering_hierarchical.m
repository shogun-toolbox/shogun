addpath('tools');
fm_train=load_matrix('../data/fm_train_real.dat');

% Hierarchical
disp('Hierarchical');

merges=3;

sg('set_features', 'TRAIN', fm_train);
sg('set_distance', 'EUCLIDEAN', 'REAL');
sg('new_clustering', 'HIERARCHICAL');

sg('train_clustering', merges);
[merge_distance, pairs]=sg('get_clustering');
