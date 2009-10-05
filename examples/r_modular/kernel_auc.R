library(shogun)

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))

# auc
#print('AUC')
#
#feats_train <- RealFeatures(fm_train_real)
#feats_test <- RealFeatures(fm_test_real)
#width <- 1.7
#subkernel <- GaussianKernel(feats_train, feats_test, width)
#
#num_feats <- 2; # do not change!
#len_train <- 11
#len_test <- 17
#data <- uint16((len_train-1)*rand(num_feats, len_train))
#feats_train <- WordFeatures(data)
#data <- uint16((len_test-1)*rand(num_feats, len_test))
#feats_test <- WordFeatures(data)
#
#kernel <- AUCKernel(feats_train, feats_test, subkernel)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
