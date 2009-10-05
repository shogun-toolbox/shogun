library(shogun)

# Explicit examples on how to use the different kernels

fm_train_word <- as.matrix(read.table('../data/fm_train_word.dat'))
fm_test_word <- as.matrix(read.table('../data/fm_test_word.dat'))

## word_match
#print('WordMatch')
#
#feats_train <- WordFeatures(fm_train_word)
#feats_test <- WordFeatures(fm_test_word)
#degree <- 3
#do_rescale <- TRUE
#scale <- 1.4
#
#kernel <- WordMatchKernel(feats_train, feats_train, degree, do_rescale, scale)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
