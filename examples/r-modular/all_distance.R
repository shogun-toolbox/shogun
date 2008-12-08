library(shogun)

# Explicit examples on how to use the different distances

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))


###########################################################################
# real features
###########################################################################

# bray curtis distance
print('BrayCurtisDistance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- BrayCurtisDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()


# euclidian distance
print('EuclidianDistance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- EuclidianDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# norm squared distance
print('EuclidianDistance - NormSquared')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- EuclidianDistance(feats_train, feats_train)
dump <- distance$set_disable_sqrt(distance,TRUE)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# canberra metric
print('CanberaMetric')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- CanberraMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# chebyshew metric
print('ChebyshewMetric')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- ChebyshewMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# chi square distance
print('ChiSquareDistance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- ChiSquareDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# cosine distance
print('CosineDistance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- CosineDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()


# geodesic metric
print('GeodesicMetric')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- GeodesicMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# jensen metric
print('JensenMetric')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- JensenMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# manhattan metric
print('ManhattanMetric')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- ManhattanMetric(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# minkowski metric
print('MinkowskiMetric')

k = 4
feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- MinkowskiMetric(feats_train, feats_train, k)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# tanimoto distance
print('TanimotoDistance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

distance <- TanimotoDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()


# sparse euclidian distance
print('SparseEuclidianDistance')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

distance <- SparseEuclidianDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()


###########################################################################
# complex string features
############################################################################

# canberra word distance
print('CanberraWordDistance')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

distance <- CanberraWordDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# hamming word distance
print('HammingWordDistance')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

distance <- HammingWordDistance(feats_train, feats_train, use_sign)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()

# manhattan word distance
print('ManhattanWordDistance')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

distance <- ManhattanWordDistance(feats_train, feats_train)

dm_train <- distance$get_distance_matrix()
dump <- distance$init(distance, feats_train, feats_test)
dm_test <- distance$get_distance_matrix()
