dyn.load('features/Features.so')
dyn.load('distance/Distance.so')
source('features/Features.R')
source('distance/Distance.R')
cacheMetaData(1)

len <- 17
num <- 42
dist <- 2.3

# Explicit examples on how to use the different distances
acgt <- c('A', 'C', 'G', 'T')
trainlab <- c(rep(1,num/2),rep(-1,num/2))
traindata_dna <- list()
testdata_dna <- list()
for (i in 1:num)
{
	traindata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep <- "", collapse <- "")
	testdata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep <- "", collapse <- "")
}

trainlab <- c(rep(-1,num/2), rep(1,num/2))
testlab <- c(rep(-1,num/2), rep(1,num/2))
traindata_real <- matrix(c(rnorm(num)-dist,rnorm(num)+dist),2,num)
testdata_real <- matrix(c(rnorm(num)-dist,rnorm(num)+dist),2,num)
###########################################################################
# real features
###########################################################################

# euclidian distance
print('EuclidianDistance')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- EuclidianDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# norm squared distance
print('EuclidianDistance - NormSquared')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- EuclidianDistance(feats_train, feats_train)
distance$set_disable_sqrt(distance,TRUE)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# canberra metric
print('CanberaMetric')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- CanberraMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# chebyshew metric
print('ChebyshewMetric')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- ChebyshewMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# geodesic metric
print('GeodesicMetric')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- GeodesicMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# jensen metric
print('JensenMetric')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- JensenMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# manhattan metric
print('ManhattanMetric')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- ManhattanMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# minkowski metric
print('MinkowskiMetric')

k = 4
feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

distance <- MinkowskiMetric(feats_train, feats_train, k)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# sparse euclidian distance
print('SparseEuclidianDistance')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

distance <- SparseEuclidianDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()


###########################################################################
# complex string features
############################################################################
#
## canberra word distance
#print('CanberraWordDistance')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#distance <- CanberraWordDistance(feats_train, feats_train)
#
#dm_train <- distance$get_distance_matrix()
#distance$init(distance, feats_train, feats_test)
#dm_test <- distance$get_distance_matrix()
#
## hamming word distance
#print('HammingWordDistance')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#use_sign <- FALSE
#
#distance <- HammingWordDistance(feats_train, feats_train, use_sign)
#
#dm_train <- distance$get_distance_matrix()
#distance$init(distance, feats_train, feats_test)
#dm_test <- distance$get_distance_matrix()
#
## manhattan word distance
#print('ManhattanWordDistance')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#distance <- ManhattanWordDistance(feats_train, feats_train)
#
#dm_train <- distance$get_distance_matrix()
#distance$init(distance, feats_train, feats_test)
#dm_test <- distance$get_distance_matrix()
