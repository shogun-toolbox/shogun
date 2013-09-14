library(shogun)

fm_train_word <- as.matrix(read.table('../data/fm_train_word.dat'))
fm_test_word <- as.matrix(read.table('../data/fm_test_word.dat'))

## linear_word
#print('LinearWord')
#
#feats_train <- WordFeatures(fm_train_word)
#feats_test <- WordFeatures(fm_test_word)
#do_rescale <- TRUE
#scale <- 1.4
#
#kernel <- LinearWordKernel(feats_train, feats_train, do_rescale, scale)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
