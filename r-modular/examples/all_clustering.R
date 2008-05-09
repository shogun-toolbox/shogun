dyn.load('features/Features.so')
dyn.load('distance/Distance.so')
dyn.load('clustering/Clustering.so')
load('features/Features.RData')
cacheMetaData(1)
load('distance/Distance.RData')
cacheMetaData(1)
load('clustering/Clustering.RData')
cacheMetaData(1)

#source('features/Features.R')
#source('distance/Distance.R')
#source('clustering/Clustering.R')
#cacheMetaData(1)

# Explicit examples on how to use clustering

# 4 clusters
num <- 50
dist <- 2.2
len <- 2

data <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
label <- c(rep(-1,num),rep(1,num))


# KMeans
print('KMeans')

k <- as.integer(4)
feats_train <- RealFeatures(data)
feats_test <- RealFeatures(data)
distance <- EuclidianDistance(feats_train, feats_train)

kmeans <- KMeans(k, distance)
kmeans$train()

distance$init(distance, feats_train, feats_test)
c <- kmeans$get_cluster_centers()
r <- kmeans$get_radiuses()

# Hierarchical
print('Hierarchical')

merges <- as.integer(4)
feats_train <- RealFeatures(data)
feats_test <- RealFeatures(data)
distance <- EuclidianDistance(feats_train, feats_train)

hierarchical <- Hierarchical(merges, distance)
hierarchical$train()

distance$init(distance, feats_train, feats_test)
mdist <- hierarchical$get_merge_distances()
pairs <- hierarchical$get_cluster_pairs()
