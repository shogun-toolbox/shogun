library(shogun)

fm_train_word <- as.matrix(read.table('../data/fm_train_word.dat'))
fm_test_word <- as.matrix(read.table('../data/fm_test_word.dat'))

## poly_match_word
#print('PolyMatchWord')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#degree <- 2
#inhomogene <- TRUE
#
#kernel <- PolyMatchWordKernel(feats_train, feats_train, degree, inhomogene)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
