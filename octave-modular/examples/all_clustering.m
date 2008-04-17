init_shogun

% Explicit examples on how to use clustering

% 4 clusters
num=50;
dist=2.2;

data=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
label=[ones(1,num) 2*ones(1,num) 3*ones(1,num) 4*ones(1,num)];


% KMeans
disp('KMeans')

k=4;
feats_train=RealFeatures(data);
feats_test=RealFeatures(data);
distance=EuclidianDistance(feats_train, feats_train);

kmeans=KMeans(k, distance);
kmeans.train();

distance.init(feats_train, feats_test);
c=kmeans.get_cluster_centers();
r=kmeans.get_radiuses();

% Hierarchical
disp('Hierarchical')

merges=4;
feats_train=RealFeatures(data);
feats_test=RealFeatures(data);
distance=EuclidianDistance(feats_train, feats_train);

hierarchical=Hierarchical(merges, distance);
hierarchical.train();

distance.init(feats_train, feats_test);
mdist=hierarchical.get_merge_distances();
pairs=hierarchical.get_cluster_pairs();
