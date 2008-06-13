library(shogun)

# Explicit examples on how to use clustering

fm_train <- as.matrix(read.table('../data/fm_train_real.dat'))


# KMeans
print('KMeans')

k <- as.integer(4)
feats_train <- RealFeatures(fm_train)
feats_test <- RealFeatures(fm_train)
distance <- EuclidianDistance(feats_train, feats_train)

kmeans <- KMeans(k, distance)
kmeans$train()

distance$init(distance, feats_train, feats_test)
c <- kmeans$get_cluster_centers()
r <- kmeans$get_radiuses()

# Hierarchical
print('Hierarchical')

merges <- as.integer(4)
feats_train <- RealFeatures(fm_train)
feats_test <- RealFeatures(fm_train)
distance <- EuclidianDistance(feats_train, feats_train)

hierarchical <- Hierarchical(merges, distance)
hierarchical$train()

distance$init(distance, feats_train, feats_test)
mdist <- hierarchical$get_merge_distances()
pairs <- hierarchical$get_cluster_pairs()
