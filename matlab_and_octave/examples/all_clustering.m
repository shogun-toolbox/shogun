% Explicit examples on how to use clustering

num=10;
dist=2.2;


% KMEANS
disp('KMeans');

k=3;
iter=1000;

traindata=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
trainlab=[ones(1,num) 2*ones(1,num) 3*ones(1,num) 4*ones(1,num)];

sg('set_features', 'TRAIN', traindata);
sg('set_labels', 'TRAIN', trainlab);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('init_distance', 'TRAIN');
sg('new_clustering', 'KMEANS');
sg('train_clustering', k, iter);
[radi, centers]=sg('get_clustering');


% Hierarchical
disp('Hierarchical');

merges=3;
dist=10;
dims=2;

traindata=[randn(dims,num)-dist, randn(dims,num)+dist, randn(dims,num)+dist*[ones(dims/2,num); zeros(dims/2,num)], randn(dims,num)+dist*[zeros(dims/2,num); ones(dims/2,num)]];
i=randperm(size(traindata,2));
traindata=traindata(:,i);

sg('set_features', 'TRAIN', traindata);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('init_distance', 'TRAIN');
sg('new_clustering', 'HIERARCHICAL');
sg('train_clustering', merges);

[merge_distance, pairs]=sg('get_clustering');
