dyn.load('features/Features.so')
dyn.load('preproc/PreProc.so')
dyn.load('kernel/Kernel.so')
load('features/Features.RData')
cacheMetaData(1)
load('preproc/PreProc.RData')
cacheMetaData(1)
load('kernel/Kernel.RData')
cacheMetaData(1)

#source('features/Features.R')
#source('preproc/PreProc.R')
#source('kernel/Kernel.R')
#cacheMetaData(1)

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
preproc$init(preproc, feats_train)
feats_train$add_preproc(feats_train, preproc)
feats_train$apply_preproc(feats_train)
feats_test$add_preproc(feats_test, preproc)
feats_test$apply_preproc(feats_train)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#NormOne
print('NormOne')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- NormOne()
preproc$init(preproc, feats_train)
feats_train$add_preproc(feats_train, preproc)
feats_train$apply_preproc(feats_train)
feats_test$add_preproc(feats_test, preproc)
feats_test$apply_preproc(feats_test)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#PruneVarSubMean
print('PruneVarSubMean')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- PruneVarSubMean()
preproc$init(preproc, feats_train)
feats_train$add_preproc(feats_train, preproc)
feats_train$apply_preproc(feats_train)
feats_test$add_preproc(feats_test, preproc)
feats_test$apply_preproc(feats_test)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# word features
###########################################################################

#LinearWord - unsigned 16bit type not supported by R
#print('LinearWord')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#
#preproc <- SortWord()
#preproc$init(preproc, feats_train)
#feats_train$add_preproc(feats_train, preproc)
#feats_train$apply_preproc(feats_train)
#feats_test$add_preproc(feats_test, preproc)
#feats_test$apply_preproc(feats_test)
#
#do_rescale <- TRUE
#scale <- 1.4
#
#kernel <- LinearWordKernel(feats_train, feats_train, do_rescale, scale)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()

###########################################################################
# complex string features
###########################################################################

#CommWordString
print('CommWordString')

order <- 3
gap <- 0
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
feats_train$obtain_from_char(feats_train, charfeat, order-1, order, gap, reverse)
preproc <- SortWordString()
preproc$init(preproc, feats_train)
feats_train$add_preproc(feats_train, preproc)
feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
feats_test$obtain_from_char(feats_test, charfeat, order-1, order, gap, reverse)
feats_test$add_preproc(feats_test, preproc)
feats_test$apply_preproc(feats_test)

use_sign <- FALSE
normalization <- "FULL_NORMALIZATION"

kernel <- CommWordStringKernel(
	feats_train, feats_train, use_sign, normalization)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

#CommUlongString
print('CommUlongString')

order <- 3
gap <- 0
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringUlongFeatures(charfeat$get_alphabet())
feats_train$obtain_from_char(feats_train, charfeat, order-1, order, gap, reverse)

charfeat <- StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringUlongFeatures(charfeat$get_alphabet())
feats_test$obtain_from_char(feats_test, charfeat, order-1, order, gap, reverse)

preproc <- SortUlongString()
preproc$init(preproc, feats_train)
feats_train$add_preproc(feats_train, preproc)
feats_train$apply_preproc(feats_train)
feats_test$add_preproc(feats_test, preproc)
feats_test$apply_preproc(feats_test)

use_sign <- FALSE
normalization <- "FULL_NORMALIZATION"

kernel <- CommUlongStringKernel(
	feats_train, feats_train, use_sign, normalization)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
