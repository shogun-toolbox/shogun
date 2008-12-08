library(shogun)

# Explicit examples on how to use the different classifiers

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))


#LogPlusOne
print('LogPlusOne')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- LogPlusOne()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_train)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#NormOne
print('NormOne')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- NormOne()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#PruneVarSubMean
print('PruneVarSubMean')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- PruneVarSubMean()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# complex string features
###########################################################################

#CommWordString
print('CommWordString')

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

kernel <- CommWordStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#CommUlongString
print('CommUlongString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)

preproc <- SortUlongString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

kernel <- CommUlongStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
